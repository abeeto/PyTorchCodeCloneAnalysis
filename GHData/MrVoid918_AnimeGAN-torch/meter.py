import torch


class AverageMeter(object):
    """Computes and stores the average and current value.
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L363"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class LossMeters:
    """Wrapping meters for different losses in a class"""

    def __init__(self, *args: str):
        for arg in args:
            setattr(self, arg, AverageMeter(arg))

    def update(self, *args: str):
        assert len(self.__dict__) == len(args)
        for i in range(len(args)):
            list(self.__dict__.values())[i].update(args[i])

    def reset(self):
        for meter in list(self.__dict__.values()):
            meter.reset()

    def as_dict(self, attr: str):
        dict_object = next(iter(self.__dict__.values()))
        assert hasattr(dict_object, attr), "Has no input attribute"
        if hasattr(getattr(dict_object, attr), 'device'):
            '''Hacky way to check if meter is updated with torch tensors, since all
            torch tensors have device attribute, hence can only update values with scalar
            or torch tensors'''
            return {key: getattr(value, attr).item() for key, value in self.__dict__.items()}

        return {key: getattr(value, attr) for key, value in self.__dict__.items()}
