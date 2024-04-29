import torch as th

def accuracy(pred, labels):
    return th.sum(pred.argmax(dim=1) == labels) / len(pred)
