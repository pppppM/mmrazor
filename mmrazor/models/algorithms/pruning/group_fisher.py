# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from mmengine import MessageHub, MMLogger, Config
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from mmengine.logging import print_log

from mmrazor.models.mutables import MutableChannelUnit
from mmrazor.models.mutators import ChannelMutator
from mmrazor.models.task_modules import RecorderManager, ModuleInputsRecorder
from mmrazor.registry import MODELS
from mmengine.runner import save_checkpoint
from ..base import BaseAlgorithm

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class GroupFisher(BaseAlgorithm):
    """ItePruneAlgorithm prunes a model iteratively until reaching a prune-
    target.

    Args:
        architecture (Union[BaseModel, Dict]): The model to be pruned.
        mutator_cfg (Union[Dict, ChannelMutator], optional): The config
            of a mutator. Defaults to dict( type='ChannelMutator',
            channel_unit_cfg=dict( type='SequentialMutableChannelUnit')).
        data_preprocessor (Optional[Union[Dict, nn.Module]], optional):
            Defaults to None.
        target_pruning_ratio (dict, optional): The prune-target. The template
            of the prune-target can be get by calling
            mutator.choice_template(). Defaults to {}.
        step_freq (int, optional): The step between two pruning operations.
            Defaults to 1.
        prune_times (int, optional): The total times to prune a model.
            Defaults to 1.    module.register_forward_hook(self.save_input_forward_hook)
                module.register_backward_hook(self.compute_fisher_backward_hook)
            Defaults to True.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: Union[Dict, ChannelMutator] = dict(
                     type='ChannelMutator',
                     channel_unit_cfg=dict(
                         type='SequentialMutableChannelUnit')),
                 delta='acts',
                 interval=10,
                 batch_size=16,
                 save_ckpt_delta_thr = [1.0,0.5,0.25],
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(architecture, data_preprocessor, init_cfg)
        # using sync bn or normal bn
        import torch.distributed as dist
        if dist.is_initialized():
            print_log('Convert Bn to SyncBn.')
            self.architecture = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.architecture)
        else:
            from mmengine.model import revert_sync_batchnorm
            self.architecture = revert_sync_batchnorm(self.architecture)

        self.cur_iter = 0
        self.interval = interval
        self.batch_size = batch_size
        self.save_ckpt_delta_thr = save_ckpt_delta_thr

        # mutator
        self.mutator: ChannelMutator = MODELS.build(mutator)
        self.mutator.prepare_from_supernet(self.architecture)

        self.module_dict = dict(self.architecture.named_modules())
        
        self.conv_names = self._map_conv_and_module_name(self.module_dict)

        self.delta = delta
        # The key of self.input is conv module, and value of it
        # is list of conv' input_features in forward process
        self.conv_inputs = {}
        # The key of self.flops is conv module, and value of it
        # is the summation of conv's flops in forward process
        self.flops = {}
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts = {}
        # The key of self.temp_fisher_info is conv module, and value
        # is a temporary variable used to estimate fisher.
        self.temp_fisher_info = {}

        # The key of self.batch_fishers is conv module, and value
        # is the estimation of fisher by single batch.
        self.batch_fishers = {}

        # The key of self.accum_fishers is conv module, and value
        # is the estimation of parameter's fisher by all the batch
        # during number of self.interval iterations.
        self.accum_fishers = {}
        self.channels = 0
        self.delta = delta
        

        for conv, name in self.conv_names.items():
            self.conv_inputs[conv] = []
            self.temp_fisher_info[conv] = conv.weight.data.new_zeros(
                self.batch_size, conv.in_channels)
            self.accum_fishers[conv] = conv.weight.data.new_zeros(
                conv.in_channels)
        for unit in self.mutator.units:
            group = unit.name
            # module = self.groups[group_id][0]
            self.temp_fisher_info[group] = conv.weight.data.new_zeros(
                self.batch_size, unit.mutable_channel.num_channels)
            self.accum_fishers[group] = conv.weight.data.new_zeros(
                unit.mutable_channel.num_channels)
        self._register_hooks(self.module_dict)
        self.init_flops_acts()
        self.init_temp_fishers()

        
    def _map_conv_and_module_name(self, named_modules):

        conv2name = OrderedDict()
        for unit in self.mutator.units:
            for in_channel in unit.input_related:
                src_module = in_channel.name
                if src_module not in named_modules:
                    continue
                module = named_modules[src_module]
                

                if isinstance(module, nn.Conv2d):
                    conv2name[module] = src_module

            for out_channel in unit.output_related:
                src_module = out_channel.name
                if src_module not in named_modules:
                    continue
                module = named_modules[src_module]
                

                if isinstance(module, nn.Conv2d):
                    conv2name[module] = src_module
        return conv2name

    def _register_hooks(self, named_modules):
        """Register forward and backward hook to Conv module."""
        for unit in self.mutator.units:
            for in_channel in unit.input_related:
                src_module = in_channel.name

                if src_module not in named_modules:
                    continue

                module = named_modules[src_module]

                if not isinstance(module, nn.Conv2d):
                    continue
                module.register_forward_hook(self._save_input_forward_hook)
                module.register_backward_hook(self._compute_fisher_backward_hook)


            

    def _save_input_forward_hook(self, module, inputs, outputs):
        """Save the input and flops and acts for computing fisher and flops or
        acts.
        Args:
            module (nn.Module): the module of register hook
            inputs (tuple): input of module
            outputs (tuple): out of module
        """

        n, oc, oh, ow = outputs.shape
        ic = module.in_channels // module.groups
        kh, kw = module.kernel_size
        self.flops[module] += np.prod([n, oc, oh, ow, ic, kh, kw])
        self.acts[module] += np.prod([n, oc, oh, ow])
        # a conv module may has several inputs in graph,for example
        # head in Retinanet
        if inputs[0].requires_grad:
            self.conv_inputs[module].append(inputs)

    def _compute_fisher_backward_hook(self, module, grad_input, *args):
        """
        Args:
            module (nn.Module): module register hooks
            grad_input (tuple): tuple contains grad of input and parameters,
                grad_input[0]is the grad of input in Pytorch 1.3, it seems
                has changed in Higher version
        """
        def compute_fisher(input, grad_input):
            grads = input * grad_input
            grads = grads.sum(-1).sum(-1)
            return grads

        if module in self.conv_names and grad_input[0] is not None:
            
            feature = self.conv_inputs[module].pop(-1)[0]
            grad_feature = grad_input[0]
            # avoid that last batch is't full,
            # but actually it's always full in mmdetection.

            cur_mask = module.mutable_attrs.in_channels.current_mask
            
            self.temp_fisher_info[module][:grad_input[0].size(0), cur_mask] += \
                compute_fisher(feature, grad_feature)
        

    def accumulate_fishers(self):
        """Accumulate all the fisher during self.interval iterations."""

        for module, name in self.conv_names.items():
            self.accum_fishers[module] += self.batch_fishers[module]
        for unit in self.mutator.units:
            group = unit.name
            self.accum_fishers[group] += self.batch_fishers[group]

    def reduce_fishers(self):
        """Collect fisher from all rank."""
        for module, name in self.conv_names.items():
            dist.all_reduce(self.batch_fishers[module])
        for unit in self.mutator.units:
            group = unit.name
            dist.all_reduce(self.batch_fishers[group])

    def group_fishers(self):
        """Accumulate all module.in_mask's fisher and flops in same group."""
        for unit in self.mutator.units:
            self.flops[unit.name] = 0
            self.acts[unit.name] = 0

            activated_channels = unit.mutable_channel.activated_channels
            
            for input_channel in unit.input_related:
                if input_channel.name not in self.module_dict:
                    continue
                module = self.module_dict[input_channel.name]

                if not isinstance(module, nn.Conv2d):
                    continue
               
                module_fisher = self.temp_fisher_info[module]

                self.temp_fisher_info[unit.name] += module_fisher
                delta_flops = self.flops[module] // module.in_channels // \
                    module.out_channels * activated_channels
                self.flops[unit.name] += delta_flops

            self.batch_fishers[unit.name] = (
                self.temp_fisher_info[unit.name]**2).sum(0)
            

            for output_channel in unit.output_related:
                if output_channel.name not in self.module_dict:
                    continue
                module = self.module_dict[output_channel.name]

                if not isinstance(module, nn.Conv2d):
                    continue

                
                delta_flops = self.flops[module] // module.out_channels // \
                        module.in_channels * activated_channels
                
                self.flops[unit.name] += delta_flops
                acts = self.acts[module] // module.out_channels
                self.acts[unit.name] += acts

        for module, name in self.conv_names.items():
            self.batch_fishers[module] = (
                self.temp_fisher_info[module]**2).sum(0)

    def init_flops_acts(self):
        """Clear the flops and acts of model in last iter."""
        for module, name in self.conv_names.items():
            self.flops[module] = 0
            self.acts[module] = 0

    def init_temp_fishers(self):
        """Clear fisher info of single conv and group."""
        for module, name in self.conv_names.items():
            self.temp_fisher_info[module].zero_()
        for unit in self.mutator.units:
            group = unit.name
            self.temp_fisher_info[group].zero_()



    def compute_flops_acts(self):
        """Computing the flops and activation remains."""
        flops = 0
        max_flops = 0
        acts = 0
        max_acts = 0
        for module, _ in self.conv_names.items():
            max_flop = self.flops[module]
            in_channels = module.in_channels
            out_channels = module.out_channels

            act_in_channels = module.mutable_attrs['in_channels'].activated_channels
            act_out_channels = module.mutable_attrs['out_channels'].activated_channels
            flops += max_flop / (in_channels * out_channels) * (
                act_in_channels * act_out_channels)
            max_flops += max_flop
            max_act = self.acts[module]
            acts += max_act / out_channels * act_out_channels
            max_acts += max_act
        return flops / max_flops, acts/ max_acts


    def init_accum_fishers(self):
        """Clear accumulated fisher info."""
        for module, name in self.conv_names.items():
            self.accum_fishers[module].zero_()
        for unit in self.mutator.units:
            group = unit.name
            self.accum_fishers[group].zero_()


    def channel_prune(self):
        """Select the channel in model with smallest fisher / delta set
        corresponding in_mask 0."""

        info = {'module': None, 'channel': None, 'min': 1e9}

        for unit in self.mutator.units:
            group = unit.name
            in_mask = unit.mutable_channel.current_mask
            fisher = self.accum_fishers[group].double()
            if self.delta == 'flops':
                fisher /= float(self.flops[group] / 1e9)
            elif self.delta == 'acts':
                fisher /= float(self.acts[group] / 1e6)
            
            info.update(self.find_pruning_channel(group, fisher, in_mask,
                                                  info))

        module, channel = info['module'], info['channel']
        for unit in self.mutator.units:
            group = unit.name
            if module == group:
                cur_mask = unit.mutable_channel.current_mask
                cur_mask[channel] = False
                unit.mutable_channel.current_choice = cur_mask
                
                break

        flops, acts = self.compute_flops_acts()
        print_log(f'slim {module} {channel}th channel, flops {flops:.2f}, acts {acts:.2f}')
        

    def find_pruning_channel(self, module, fisher, in_mask, info):
        """Find the the channel of a model to pruning.
        Args:
            module (nn.Conv | int ): Conv module of model or idx of self.group
            fisher(Tensor): the fisher information of module's in_mask
            in_mask (Tensor): the squeeze in_mask of modules
            info (dict): store the channel of which module need to pruning
                module: the module has channel need to pruning
                channel: the index of channel need to pruning
                min : the value of fisher / delta
        Returns:
            dict: store the current least important channel
                module: the module has channel need to be pruned
                channel: the index of channel need be to pruned
                min : the value of fisher / delta
        """
        module_info = {}
        if fisher.sum() > 0 and in_mask.sum() > 0:
            nonzero = in_mask.nonzero().view(-1)
            fisher = fisher[nonzero]
            min_value, argmin = fisher.min(dim=0)
            if min_value < info['min']:
                module_info['module'] = module
                module_info['channel'] = nonzero[argmin]
                module_info['min'] = min_value
        return module_info

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward."""

        if self.cur_iter > 0:
            self.group_fishers()
            if dist.is_initialized():
                self.reduce_fishers()
            self.accumulate_fishers()
            self.init_temp_fishers()
            
            if self.cur_iter % self.interval == 0:
                self.channel_prune()
                self.init_accum_fishers()
                flops, acts = self.compute_flops_acts()
                if self.delta == 'flops':
                    if flops < self.save_ckpt_delta_thr[0]:
                        hub = MessageHub.get_current_instance()
                        import pdb;pdb.set_trace()
                        self.save_ckpt_delta_thr.pop(0)
                else:
                    if acts < self.save_ckpt_delta_thr[0]:
                        hub = MessageHub.get_current_instance()
                        cfg = Config.fromstring(hub.runtime_info['cfg'],'.py')
                        self.save_ckpt_delta_thr.pop(0)
                        ckpt = {'state_dict': self.architecture.state_dict()}
                        save_path = f'{cfg.work_dir}/act_{acts:.2f}.pth'
                        save_checkpoint(ckpt, save_path)

                        print_log(f'Save checkpoint to {save_path}')
            self.init_flops_acts()
        self.cur_iter += 1
        

        return super().forward(inputs, data_samples, mode)


    