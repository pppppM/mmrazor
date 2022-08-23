# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, NamedTuple, Optional, TypeVar

from mmcv.runner import BaseModule

Choice = TypeVar('Choice')
RuntimeChoice = TypeVar('RuntimeChoice')
Chosen = TypeVar('Chosen')


class DumpChosen(NamedTuple):

    chosen: Chosen
    meta: Optional[Dict[str, Any]]


class BaseMutable(BaseModule, ABC, Generic[Choice, RuntimeChoice, Chosen]):
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
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.alias = alias

    @property
    @abstractmethod
    def choices(self) -> List[CHOICE_TYPE]:
        """list: all choices.  All subclasses must implement this method."""

    @property
    @abstractmethod
    def current_choice(self) -> RuntimeChoice:
        """Current choice will affect :meth:`forward` and will be used in
        :func:`mmrazor.core.subnet.utils.export_fix_subnet` or mutator.
        """

    @current_choice.setter
    @abstractmethod
    def current_choice(self, choice: Choice) -> None:
        """Current choice setter will be executed in mutator."""

    @property
    @abstractmethod
    def is_fixed(self) -> bool:
        """bool: whether the mutable is fixed.

        Note:
            If a mutable is fixed, it is no longer a searchable module, just
                a normal fixed module.
            If a mutable is not fixed, it still is a searchable module.
        """
        return self._is_fixed

    @is_fixed.setter
    @abstractmethod
    def is_fixed(self, is_fixed: bool) -> None:
        """Set the status of `is_fixed`."""

    @abstractmethod
    def fix_chosen(self, chosen: Chosen) -> None:
        """Fix mutable with choice. This function would fix the choice of
        Mutable. The :attr:`is_fixed` will be set to True and only the selected
        operations can be retained. All subclasses must implement this method.

        Note:
            This operation is irreversible.
        """

    # TODO
    # type hint
    @abstractmethod
    def dump_chosen(self) -> DumpChosen:
        pass
