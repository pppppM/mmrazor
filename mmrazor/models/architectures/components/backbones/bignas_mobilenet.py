# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import make_divisible
from mmcv.cnn import ConvModule
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.builder import build_op
from ...utils import Placeholder


@BACKBONES.register_module()
class BigNASMobileNet(BaseBackbone):
    """Searchable MobileNet backbone.

    Args:
        first_channels (int): Channel width of first ConvModule. Default: 32.
        last_channels (int): Channel width of last ConvModule. Default: 1200.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        arch_setting_type (str): Specify architecture setting.
            Default: 'original'.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. OpenMMLab has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.
    """

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 24, 2, 1], [6, 32, 3, 2], [6, 48, 3, 2],
                     [6, 88, 4, 2], [6, 128, 6, 1], [6, 216, 6, 2],
                     [6, 352, 2, 1]]

    def __init__(self,
                 first_channels=40,
                 last_channels=1408,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super().__init__(init_cfg)

        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(first_channels * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio,
                stage_idx=i)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(last_channels * widen_factor)
        else:
            self.out_channel = last_channels

        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

    def make_layer(self, out_channels: int, num_blocks: int, stride: int,
                   expand_ratio: int, stage_idx: int) -> Module:
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1

            if stage_idx < 2:
                group = 'dynamic_conv2d_k3'
            else:
                group = 'dynamic_conv2d_k35'

            layers.append(
                build_op(
                    cfg=dict(
                        type='SearchableMBBlock',
                        group=group,
                        space_id_prefix=f'stage_{stage_idx}_block_{i}',
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        expand_ratio=expand_ratio,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        with_cp=self.with_cp)))

            self.in_channels = out_channels

        stage = nn.Sequential(*layers)

        return Placeholder(
            group=f'stage{stage_idx}',
            space_id=f'stage_{stage_idx}',
            choices=stage)

    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        """Freeze params not to update in the specified stages."""
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
