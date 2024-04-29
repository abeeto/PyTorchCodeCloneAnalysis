class AverageMeter():
    """Meter for monitoring losses during training. """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average by val and n, where val is the avarage of n values. """
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
