# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmcv.cnn import MODELS
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.builder import ALGORITHMS, build_loss


@ALGORITHMS.register_module()
class SingleStageDetNeckDistill(BaseModule):
    """An example for Single Stage Detector FPN distillation."""

    def __init__(self,
                 student,
                 teacher,
                 teacher_norm_eval=True,
                 distill_loss=dict(
                     type='ChannelWiseDivergence', loss_weight=10),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.student = MODELS.build(student)
        self.teacher = MODELS.build(teacher)
        self.teacher_norm_eval = teacher_norm_eval
        self.distill_loss = build_loss(distill_loss)

    def train(self, mode=True):
        """Set distiller's forward mode."""
        super().train(mode)
        if mode and self.teacher_norm_eval:
            for m in self.teacher.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, img, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        return self.student(img, return_loss=return_loss, **kwargs)

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        return self.student.simple_test(img, img_metas)

    def show_result(self, img, result, **kwargs):
        """Draw `result` over `img`"""
        return self.student.show_result(img, result, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        img = data['img']
        img_metas = data['img_metas']
        gt_bboxes = data['gt_bboxes']
        gt_labels = data['gt_labels']
        gt_bboxes_ignore = getattr(data, 'gt_bboxes_ignore', None)

        student_feats = self.student.extract_feat(img)
        with torch.no_grad():
            teacher_feats = self.teacher.extract_feat(img)

        student_losses = self.student.bbox_head.forward_train(
            student_feats, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

        distill_losses = dict()
        for i, (s_feat,
                t_feat) in enumerate(zip(student_feats, teacher_feats)):
            loss = self.distill_loss(s_feat, t_feat)
            distill_losses[f'distill_loss.{i}'] = loss

        losses = dict()
        losses.update(student_losses)
        losses.update(distill_losses)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs
