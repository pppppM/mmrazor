# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ..base_mutable import (BaseMutable, Choice, Chosen, DumpChosen,
                            RuntimeChoice)


class MutableModule(BaseMutable[str, str, str]):
    """Base Class for mutables. Mutable means a searchable module widely used
    in Neural Architecture Search(NAS).

    It mainly consists of some optional operations, and achieving
    searchable function by handling choice with ``MUTATOR``.

    All subclass should implement the following APIs:

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
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.module_kwargs = module_kwargs
        self._is_fixed = False
        self._current_choice: Optional[Choice] = None

    @property
    def current_choice(self) -> Choice:
        """Current choice will affect :meth:`forward` and will be used in
        :func:`mmrazor.core.subnet.utils.export_fix_subnet` or mutator.
        """
        assert self._current_choice is not None
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: Choice) -> None:
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

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward computation."""
