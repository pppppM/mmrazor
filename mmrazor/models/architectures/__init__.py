# Copyright (c) OpenMMLab. All rights reserved.
from .components import *  # noqa: F401,F403
from .mmcls import MMClsArchitecture
from .mmdet import MMDetArchitecture
from .mmseg import MMSegArchitecture

__all__ = ['MMClsArchitecture', 'MMDetArchitecture', 'MMSegArchitecture']
