import mmcv
from mmcv.runner import HOOKS, LrUpdaterHook

@HOOKS.register_module()
class RazorStepLrUpdaterHook(LrUpdaterHook):
    """Step LR scheduler with min_lr clipping.
    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float, optional): Decay LR ratio. Default: 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self, step, gamma=0.1, min_lr=None, 
                 per_epoch_iters=1,include_warmup_progress=False, **kwargs):
        if isinstance(step, list):
            assert mmcv.is_list_of(step, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, (int,float)):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        self.include_warmup = include_warmup_progress
        self.per_epoch_iters = per_epoch_iters
        super(RazorStepLrUpdaterHook, self).__init__(**kwargs)

        # if not self.include_warmup:
        #     assert self.warmup_by_epoch == self.by_epoch

    @property
    def warmup_progress(self):
        if self.warmup_by_epoch:
            return self.warmup_epochs * self.per_epoch_iters
        else:
            return self.warmup_iters

    def get_lr(self, runner, base_lr):

        progress = runner.iter
        if not self.include_warmup:
            progress = progress - self.warmup_progress

        # calculate exponential term
        if isinstance(self.step, (int,float)):
            if self.by_epoch:
                period =int(self.step * self.per_epoch_iters) 
            else:
                period = self.step
            exp = max(0, progress // period)
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                if self.warmup_by_epoch:
                    per_epoch_iter = self.warmup_iters / self.warmup_epochs
                    cur_epochs = cur_iters // per_epoch_iter
                    k = (1 - cur_epochs / self.warmup_epochs) * (1 - self.warmup_ratio)
                else:
                    k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                            self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)
    