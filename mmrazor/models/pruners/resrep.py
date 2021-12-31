# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from mmrazor.models.builder import PRUNERS
from .structure_pruning import StructurePruner


@PRUNERS.register_module()

class ResRepPruner(StructurePruner):
    """A random ratio pruner.

    Each layer can adjust its own width ratio randomly and independently.

    Args:
        ratios (list | tuple): Width ratio of each layer can be
            chosen from `ratios` randomly. The width ratio is the ratio between
            the number of reserved channels and that of all channels in a
            layer. For example, if `ratios` is [0.25, 0.5], there are 2 cases
            for us to choose from when we sample from a layer with 12 channels.
            One is sampling the very first 3 channels in this layer, another is
            sampling the very first 6 channels in this layer. Default to None.
    """
    # TODO update doc-string

    def __init__(self, flops_constraint, **kwargs):
        super(ResRepPruner, self).__init__(**kwargs)
        self.flops_constraint = flops_constraint

    
    def prepare_from_supernet(self, supernet):
        super().prepare_from_supernet(supernet)

        self.compactors = nn.ModuleDict()
        for name, out_mask in self.channel_spaces.items():
            channels = out_mask.size(1)
            compactor = nn.Conv2d(channels,channels,1,bias=False)
            compactor.out_mask = out_mask
            self.compactors[name] = compactor

        for module_name, group in self.module2group.items():
            module = self.name2module[module_name]
            module.pruning_group = group
            if type(module).__name__ == 'Conv2d' and module.groups==1:
                module.forward = self.modify_conv_forward(module,self.compactors)

    @staticmethod
    def modify_conv_forward(module, compactors):
        """Modify the forward method of a conv layer."""

        def modified_forward(self, feature):
            out = F.conv2d(feature, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
            compactor = compactors[self.pruning_group]
            out = compactor(out)

            return out

        return MethodType(modified_forward, module)
       

    def add_pruning_attrs(self, module, modify_forward=True):
        super().add_pruning_attrs(module,modify_forward=False)


    def sample_subnet(self):
        # TODO 
        pass
    #     """Random sample subnet by random mask.

    #     Returns:
    #         dict: Record the information to build the subnet from the supernet,
    #             its keys are the properties ``space_id`` in the pruner's search
    #             spaces, and its values are corresponding sampled out_mask.
    #     """
    #     subnet_dict = dict()
    #     for space_id, out_mask in self.channel_spaces.items():
    #         subnet_dict[space_id] = self.get_channel_mask(out_mask)
    #     return subnet_dict



