from torch.optim.lr_scheduler import _LRScheduler

class CustomExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        max_lr_decay (float): Multiplicative factor of max_lr.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, base_lr, max_lr, max_lr_decay=1., step_size_down=0, step_size_up=50, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)
        self.max_lr_decay = max_lr_decay

        self.step_size_up = float(step_size_up)
        self.step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size
        
        self.gamma = (base_lr / max_lr) ** (1/step_size_down)
        
        super(CustomExponentialLR, self).__init__(optimizer, last_epoch, verbose)
        
    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)
        
    def get_lr(self):
        
        cycle_epoch = self.last_epoch % self.total_size
        cycle = self.last_epoch // self.total_size
        
        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            max_lr = max_lr * (self.max_lr_decay**cycle)
            if cycle_epoch < self.step_size_up:
                lr = (max_lr - base_lr) / self.step_size_up * cycle_epoch + base_lr
            else:
                lr = max_lr * self.gamma ** (cycle_epoch - self.step_size_up)
                
            lrs.append(lr)
        
        return lrs
