import numpy as np


class Metric:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.loss = 0
        self.num_iters = 0

    def update(self, pred, label, loss):
        self.tp += ((pred >= 0.5) & (label == 1)).astype(np.int32).sum()
        self.tn += ((pred < 0.5) & (label == 0)).astype(np.int32).sum()
        self.fp += ((pred < 0.5) & (label == 1)).astype(np.int32).sum()
        self.fn += ((pred >= 0.5) & (label == 0)).astype(np.int32).sum()
        self.loss += loss
        self.num_iters += 1

    @property
    def loss(self):
        return self.loss / self.num_iters

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-10)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-10)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall + 1e-10)

    @property
    def fpr(self):
        return self.fp / (self.tn + self.fp + 1e-10)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

