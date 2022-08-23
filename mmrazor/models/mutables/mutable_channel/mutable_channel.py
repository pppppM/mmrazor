# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import List

import torch

from ..base_mutable import BaseMutable, Choice, Chosen, RuntimeChoice
from ..derived_mutable import DerivedMethodMixin


class MutableChannel(BaseMutable[Choice, RuntimeChoice, Chosen],
                     DerivedMethodMixin):
    """A type of ``MUTABLES`` for single path supernet such as AutoSlim. In
    single path supernet, each module only has one choice invoked at the same
    time. A path is obtained by sampling all the available choices. It is the
    base class for one shot channel mutables.

    Args:
        num_channels (int): The raw number of channels.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(**kwargs)

        self.num_channels = num_channels

    @property
    def current_choice(self) -> RuntimeChoice:
        """The current mask.

        We slice the registered parameters and buffers of a ``nn.Module``
        according to the mask of the corresponding channel mutable.
        """

    def fix_chosen(self, chosen: Chosen) -> None:
        """Fix mutable with subnet config. This operation would convert
        `unfixed` mode to `fixed` mode. The :attr:`is_fixed` will be set to
        True and only the selected operations can be retained.

        Args:
            chosen (str): The chosen key in ``MUTABLE``. Defaults to None.
        """
        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        self.is_fixed = True

    def __repr__(self):
        concat_mutable_name = [
            mutable.name for mutable in self.concat_parent_mutables
        ]
        repr_str = self.__class__.__name__
        repr_str += f'(name={self.name}, '
        repr_str += f'num_channels={self.num_channels}, '
        repr_str += f'concat_mutable_name={concat_mutable_name})'
        return repr_str
