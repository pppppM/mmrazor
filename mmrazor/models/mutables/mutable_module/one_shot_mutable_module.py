# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch.nn as nn
from torch import Tensor

from mmrazor.registry import MODELS
from ..base_mutable import (ChoiceItem, Chosen, DumpChosen, RuntimeChoice,
                            SampleChoice)
from .mutable_module import MutableModule


class OneShotMutableModule(MutableModule[str]):
    """Base class for one shot mutable module. A base type of ``MUTABLES`` for
    single path supernet such as Single Path One Shot.

    All subclass should implement the following APIs:

    - ``forward_fixed()``
    - ``forward_all()``
    - ``forward_choice()``

    Args:
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.

    Note:
        :meth:`forward_all` is called when calculating FLOPs.
    """

    def parse_chosen_from_choice(self, choice: str) -> List[str]:
        return [choice]

    def forward(self, x: Any) -> Any:
        """Calls either :func:`forward_fixed` or :func:`forward_choice`
        depending on whether :func:`is_fixed` is ``True`` and whether
        :func:`current_choice` is None.

        Note:
            :meth:`forward_fixed` is called in `fixed` mode.
            :meth:`forward_all` is called in `unfixed` mode with
                :func:`current_choice` is None.
            :meth:`forward_choice` is called in `unfixed` mode with
                :func:`current_choice` is not None.

        Args:
            x (Any): input data for forward computation.
            choice (CHOICE_TYPE, optional): the chosen key in ``MUTABLE``.

        Returns:
            Any: the result of forward
        """
        if self.is_fixed:
            return self.forward_fixed(x)
        if self.current_choice is None:
            return self.forward_all(x)
        else:
            return self.forward_choice(x, choice=self.current_choice)

    @abstractmethod
    def forward_fixed(self, x: Any) -> Any:
        """Forward with the fixed mutable.

        All subclasses must implement this method.
        """

    @abstractmethod
    def forward_all(self, x: Any) -> Any:
        """Forward all choices.

        All subclasses must implement this method.
        """

    @abstractmethod
    def forward_choice(self, x: Any, choice: RuntimeChoice) -> Any:
        """Forward with the unfixed mutable and current_choice is not None.

        All subclasses must implement this method.
        """


@MODELS.register_module()
class OneShotMutableOP(OneShotMutableModule):
    """A type of ``MUTABLES`` for single path supernet, such as Single Path One
    Shot. In single path supernet, each choice block only has one choice
    invoked at the same time. A path is obtained by sampling all the choice
    blocks.

    Args:
        candidates (dict[str, dict]): the configs for the candidate
            operations.
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.

    Examples:
        >>> import torch
        >>> from mmrazor.models.mutables import OneShotMutableOP

        >>> candidates = nn.ModuleDict({
        ...     'conv3x3': nn.Conv2d(32, 32, 3, 1, 1),
        ...     'conv5x5': nn.Conv2d(32, 32, 5, 1, 2),
        ...     'conv7x7': nn.Conv2d(32, 32, 7, 1, 3)})

        >>> input = torch.randn(1, 32, 64, 64)
        >>> op = OneShotMutableOP(candidates)

        >>> op.choices
        ['conv3x3', 'conv5x5', 'conv7x7']
        >>> op.num_choices
        3
        >>> op.is_fixed
        False

        >>> op.current_choice = 'conv3x3'
        >>> unfix_output = op.forward(input)
        >>> torch.all(unfixed_output == candidates['conv3x3'](input))
        True

        >>> op.fix_chosen('conv3x3')
        >>> fix_output = op.forward(input)
        >>> torch.all(fix_output == unfix_output)
        True

        >>> op.choices
        ['conv3x3']
        >>> op.num_choices
        1
        >>> op.is_fixed
        True
    """

    def forward_fixed(self, x: Any) -> Tensor:
        """Forward with the `fixed` mutable.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward the fixed operation.
        """
        assert len(self.chosen) == 1
        return self.candidates[self.chosen[0]](x)

    def forward_choice(self, x: Any, choice: RuntimeChoice) -> Tensor:
        """Forward with the `unfixed` mutable and current choice is not None.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
            choice (str): the chosen key in ``OneShotMutableOP``.

        Returns:
            Tensor: the result of forward the ``choice`` operation.
        """
        assert isinstance(choice, str) and choice in self.choices
        return self.candidates[choice](x)

    def forward_all(self, x: Any) -> Tensor:
        """Forward all choices. Used to calculate FLOPs.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward all of the ``choice`` operation.
        """
        outputs = list()
        for op in self.candidates.values():
            outputs.append(op(x))
        return sum(outputs)
