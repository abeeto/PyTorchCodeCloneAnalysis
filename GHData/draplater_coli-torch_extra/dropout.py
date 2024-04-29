import torch
from torch import Tensor, nn
from torch.autograd.function import InplaceFunction
from torch.jit import ScriptModule, script_method
from torch.nn import Dropout


class FeatureDropout2(ScriptModule):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.dropout = Dropout(p)

    @script_method
    def forward(self, input: Tensor):
        noise = self.dropout(torch.ones(input.shape[0], input.shape[2],
                                        device=input.device)
                             ).unsqueeze(-2).expand(-1, input.shape[1], -1)
        return input * noise

    @classmethod
    def load_func(cls, p, inplace):
        return cls(p, inplace)

    def __reduce__(self):
        return self.__class__.load_func, (self.p, self.inplace)


class FeatureDropoutFunction(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        batch_size, max_sent_length, feature_count = input.shape
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_size, 1, feature_count)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise.repeat(1, max_sent_length, 1)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None, None
        else:
            return grad_output, None, None, None, None


class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return FeatureDropoutFunction.apply(input, self.p, self.training, self.inplace)
