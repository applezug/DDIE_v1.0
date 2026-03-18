import math
from torch import inf
from torch.optim.optimizer import Optimizer


class ReduceLROnPlateauWithWarmup(object):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, warmup_lr=None, warmup=0):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        self.optimizer = optimizer
        self.min_lrs = list(min_lr) if isinstance(min_lr, (list, tuple)) else [min_lr] * len(optimizer.param_groups)
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.warmup_lr = warmup_lr
        self.warmup = warmup
        self.eps = eps
        self.last_epoch = 0
        self.mode_worse = inf if mode == 'min' else -inf
        self._prepare_for_warmup()
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def _prepare_for_warmup(self):
        if self.warmup_lr is not None:
            self.warmup_lrs = [self.warmup_lr] * len(self.optimizer.param_groups) if not isinstance(self.warmup_lr, (list, tuple)) else list(self.warmup_lr)
            curr_lrs = [g['lr'] for g in self.optimizer.param_groups]
            self.warmup_lr_steps = [max(0, (self.warmup_lrs[i] - curr_lrs[i]) / float(self.warmup)) for i in range(len(curr_lrs))]
        else:
            self.warmup_lrs = self.warmup_lr_steps = None

    def step(self, metrics):
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch <= self.warmup and self.warmup_lr_steps:
            for i, g in enumerate(self.optimizer.param_groups):
                g['lr'] = max(g['lr'] + self.warmup_lr_steps[i], self.min_lrs[i])
        else:
            rel = 1.0 - self.threshold if self.mode == 'min' else 1.0 + self.threshold
            better = current < self.best * rel if self.mode == 'min' else current > self.best * rel
            if better:
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0
            if self.num_bad_epochs > self.patience:
                for i, g in enumerate(self.optimizer.param_groups):
                    g['lr'] = max(g['lr'] * self.factor, self.min_lrs[i])
                self.cooldown_counter = getattr(self, 'cooldown', 0)
                self.num_bad_epochs = 0
