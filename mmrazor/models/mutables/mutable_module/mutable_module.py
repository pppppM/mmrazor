# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, overload

from torch import nn

from mmrazor.registry import MODELS
from ..base_mutable import (BaseMutable, ChoiceItem, Choices, Chosen,
                            DumpChosen, RuntimeChoice, SampleChoice)


class MutableModule(BaseMutable[str, SampleChoice, SampleChoice]):
    """Base Class for mutables. Mutable means a searchable module widely used
    in Neural Architecture Search(NAS).

    It mainly consists of some optional operations, and achieving
    searchable function by handling choice with ``MUTATOR``.

    All subclass should implement the following APIs:Optional[Dict[str, Any]] = None

    - ``forward()``
    - ``fix_chosen()``
    - ``choices()``

    Args:
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 candidates: Union[Dict[str, Dict], nn.ModuleDict],
                 candidate_probs: Optional[Dict[str, float]] = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 num_chosen: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.module_kwargs = module_kwargs
        self._is_fixed = False
        self._current_choice: Optional[SampleChoice] = None
        self._num_chosen = num_chosen

        assert len(candidates) >= 1, \
            f'Number of candidate op must greater than 1, ' \
            f'but got: {len(candidates)}'

        if isinstance(candidates, dict):
            self.candidates = self._build_modules(candidates,
                                                  self.module_kwargs)
        elif isinstance(candidates, nn.ModuleDict):
            self.candidates = candidates
        else:
            raise TypeError('candidata_ops should be a `dict` or '
                            f'`nn.ModuleDict` instance, but got '
                            f'{type(candidates)}')

        assert len(self.candidates) >= 1, \
            f'Number of candidate op must greater than or equal to 1, ' \
            f'but got {len(self.candidates)}'

        self._candidate_probs = candidate_probs

    @property
    def choice_probs(self):
        return self._candidate_probs.values()

    @property
    def num_chosen(self) -> int:
        return self._num_chosen

    @staticmethod
    def _build_modules(
            candidates: Union[Dict[str, Dict], nn.ModuleDict],
            module_kwargs: Optional[Dict[str, Dict]] = None) -> nn.ModuleDict:
        """Build candidate operations based on choice configures.

        Args:
            candidates (dict[str, dict] | :obj:`nn.ModuleDict`): the configs
                for the candidate operations or nn.ModuleDict.
            module_kwargs (dict[str, dict], optional): Module initialization
                named arguments.

        Returns:
            ModuleDict (dict[str, Any], optional):  the key of ``ops`` is
                the name of each choice in configs and the value of ``ops``
                is the corresponding candidate operation.
        """
        if isinstance(candidates, nn.ModuleDict):
            return candidates

        ops = nn.ModuleDict()
        for name, op_cfg in candidates.items():
            assert name not in ops
            if module_kwargs is not None:
                op_cfg.update(module_kwargs)
            ops[name] = MODELS.build(op_cfg)
        return ops

    @property
    def choices(self) -> List[ChoiceItem]:
        """list: all choices. """
        return list(self.candidates.keys())

    @property
    def current_choice(self) -> SampleChoice:
        """Current choice will affect :meth:`forward` and will be used in
        :func:`mmrazor.core.subnet.utils.export_fix_subnet` or mutator.
        """
        assert self._current_choice is not None
        return self._current_choice

    @BaseMutable.current_choice.setter
    def current_choice(self, choice: SampleChoice) -> None:
        """Current choice setter will be executed in mutator."""
        self._current_choice = choice

    @property
    def is_fixed(self) -> bool:
        """bool: whether the mutable is fixed.

        Note:
            If a mutable is fixed, it is no longer a searchable module, just
                a normal fixed module.
            If a mutable is not fixed, it still is a searchable module.
        """
        return self._is_fixed

    @is_fixed.setter
    def is_fixed(self, is_fixed: bool) -> None:
        """Set the status of `is_fixed`."""
        assert isinstance(is_fixed, bool), \
            f'The type of `is_fixed` need to be bool type, ' \
            f'but got: {type(is_fixed)}'
        if self._is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not set `is_fixed` function repeatedly.')
        self._is_fixed = is_fixed

    def fix_chosen(self, chosen: Chosen) -> None:
        """Fix mutable with `choice`. This operation would convert `unfixed`
        mode to `fixed` mode. The :attr:`is_fixed` will be set to True and only
        the selected operations can be retained.

        Args:
            chosen (str): the chosen key in ``MUTABLE``.
                Defaults to None.
        """
        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        if isinstance(chosen, str):
            chosen = [chosen]

        for c in self.choices:
            if c not in chosen:
                self.candidates.pop(c)

        self.is_fixed = True

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward computation."""
