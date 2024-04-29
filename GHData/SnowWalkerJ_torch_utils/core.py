import torch


USE_CUDA = torch.cuda.is_available()


class Variable(torch.autograd.Variable):
    """
    Automatically convert Tensor to its cuda version if USE_CUDA is True
    """
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            self.data = data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)