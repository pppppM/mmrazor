# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import numpy as np
import torch

from mmrazor.registry import MODELS
from .mutable_channel import MutableChannel


@MODELS.register_module()
class OneShotMutableChannel(MutableChannel[int, int]):
    """A type of ``MUTABLES`` for single path supernet such as AutoSlim. In
    single path supernet, each module only has one choice invoked at the same
    time. A path is obtained by sampling all the available choices. It is the
    base class for one shot mutable channel.

    Args:
        num_channels (int): The raw number of channels.
        candidate_choices (List): If `candidate_mode` is "ratio",
            candidate_choices is a list of candidate width ratios. If
            `candidate_mode` is "number", candidate_choices is a list of
            candidate channel number. We note that the width ratio is the ratio
            between the number of reserved channels and that of all channels in
            a layer.
            For example, if `ratios` is [0.25, 0.5], there are 2 cases
            for us to choose from when we sample from a layer with 12 channels.
            One is sampling the very first 3 channels in this layer, another is
            sampling the very first 6 channels in this layer.
        candidate_mode (str): One of "ratio" or "number".
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 num_channels: int,
                 candidate_choices: List,
                 candidate_mode: str = 'ratio',
                 init_cfg: Optional[Dict] = None):
        super(OneShotMutableChannel, self).__init__(
            num_channels=num_channels, init_cfg=init_cfg)

        self._current_choice = num_channels
        assert len(candidate_choices) > 0, \
            f'Number of candidate choices must be greater than 0, ' \
            f'but got: {len(candidate_choices)}'
        self._candidate_choices = candidate_choices
        assert candidate_mode in ['ratio', 'number']
        self._candidate_mode = candidate_mode

        self._check_candidate_choices()

    def _check_candidate_choices(self):
        """Check if the input `candidate_choices` is valid."""
        if self._candidate_mode == 'number':
            assert all([num > 0 and num <= self.num_channels
                        for num in self._candidate_choices]), \
                f'The candidate channel numbers should be in ' \
                f'range(0, {self.num_channels}].'
            assert all([isinstance(num, int)
                        for num in self._candidate_choices]), \
                'Type of `candidate_choices` should be int.'
        else:
            assert all([
                ratio > 0 and ratio <= 1 for ratio in self._candidate_choices
            ]), 'The candidate ratio should be in range(0, 1].'

    def sample_choice(self) -> int:
        """Sample an arbitrary selection from candidate choices.

        Returns:
            int: The chosen number of channels.
        """
        assert len(self.concat_parent_mutables) == 0
        num_channels = np.random.choice(self.choices)
        assert num_channels > 0, \
            f'Sampled number of channels in `Mutable` {self.name}' \
            f' should be a positive integer.'
        return num_channels

    @property
    def min_choice(self) -> int:
        """Minimum number of channels."""
        assert len(self.concat_parent_mutables) == 0
        min_channels = min(self.choices)
        assert min_channels > 0, \
            f'Minimum number of channels in `Mutable` {self.name}' \
            f' should be a positive integer.'
        return min_channels

    @property
    def max_choice(self) -> int:
        """Maximum number of channels."""
        return max(self.choices)

    @property
    def current_choice(self):
        """The current choice of the mutable."""
        assert len(self.concat_parent_mutables) == 0
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: int):
        """Set the current choice of the mutable."""
        assert choice in self.choices
        self._current_choice = choice

    @property
    def choices(self) -> List[int]:
        """list: all choices. """
        if self._candidate_mode == 'number':
            return self._candidate_choices
        candidate_choices = [
            round(ratio * self.num_channels)
            for ratio in self._candidate_choices
        ]
        return candidate_choices

    @property
    def num_choices(self) -> int:
        return len(self.choices)

    def convert_choice_to_mask(self, choice: int) -> torch.Tensor:
        """Get the mask according to the input choice."""
        num_channels = choice
        mask = torch.zeros(self.num_channels).bool()
        mask[:num_channels] = True
        return mask

    def dump_chosen(self) -> int:
        return self.current_choice