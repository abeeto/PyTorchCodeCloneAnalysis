import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Add(nn.Module):
    def forward(self, x, y):
        return x + y

    def __str__(self):
        return "Add()"


class LambdaModule(nn.Module):
    """
    Wraps a normal operation as a Module, for visualization purpose
    """
    def __init__(self, lambda_fn, repr="Lambda()"):
        super(LambdaModule, self).__init__()
        self.lambda_fn = lambda_fn
        self.repr = repr

    def forward(self, *args, **kwargs):
        return self.lambda_fn(*args, **kwargs)

    def __str__(self):
        return self.repr


class GroupConnect(nn.Module):
    def __init__(self, config, bias=True):
        super(GroupConnect, self).__init__()
        self.module_list = nn.ModuleList()
        self.repr = []
        for name, (c_in, c_out) in config:
            module = nn.Linear(c_in, c_out, bias=bias)
            self.module_list.add_module(name, module)
            self.repr.append("{}({}=>{})".format(name, c_in, c_out))
        self.repr = "\n".join(self.repr)
    
    def forward(self, x):
        y = []
        i = 0
        for name, module in self.module_list.named_children():
            c_in = module.in_features
            sub_x = x[:, i:i+c_in]
            sub_y = module(sub_x)
            y.append(sub_y)
            i += c_in
        y = torch.cat(y, 1)
        return y

    def __str__(self):
        return self.repr


class Attention(nn.Module):
    def __init__(self, input_size, output_size, activation="relu"):
        super(Attention, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(input_size, output_size*2)
        self.activation_fn = getattr(F, activation)

    def reset_parameters(self):
        self.linear.bias.data[:self.linear.out_features//2].zero_()
        self.linear.bias.data[self.linear.out_features//2:].fill_(1.0)
        init.xavier_normal(self.linear.weight.data[:self.linear.out_features//2], gain=init.calculate_gain(self.activation))
        init.xavier_normal(self.linear.weight.data[self.linear.out_features//2:], gain=1)

    def forward(self, x):
        x = self.linear(x)
        x, att = x.chunk(2, dim=1)
        att = F.sigmoid(att)
        x = self.activation_fn(x)
        return x * att


class DropConnect(nn.Module):
    """
    DropConnect randomly sets the weights of the connection to zero at a probability of `drop_prob`
    """
    def __init__(self, in_features, out_features, drop_prob=0.5, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.drop_prob = drop_prob
        super(DropConnect, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal(self.weight.data)

    def forward(self, x):
        if self.training:
            batch_size = x.size(0)
            drop_mask = torch.Tensor(batch_size, self.in_features, self.out_features).fill_(self.drop_prob).bernoulli_()
            weight = self.weight.unsqueeze(0).expand(batch_size, -1, -1) * drop_mask
            x = x.unsqueeze(1)
            y = torch.baddbmm(self.bias, x, weight).squeeze(1)
        else:
            y = torch.addmm(self.bias.squeeze(0), x, self.weight, alpha=self.drop_prob)
        return y


class Reshape(nn.Module):
    """
    Wraps the reshape operation in a Module
    """
    def __init__(self, *shape):
        self.shape = shape
        super(Reshape, self).__init__()

    def forward(self, input_variable):
        return input_variable.view(*self.shape)

    def __str__(self):
        return "Reshape" + str(self.shape)


class GetItem(nn.Module):
    def __init__(self, item):
        self.item = item
        super(GetItem, self).__init__()

    def forward(self, input_variable):
        return input_variable.__getitem__(self.item)

    def __str__(self):
        return "GetItem()"


class Squeeze(nn.Module):
    def __init__(self, *dims):
        self.dims = dims
        super(Squeeze, self).__init__()

    def forward(self, input_variable):
        return input_variable.squeeze(*self.dims)

    def __str__(self):
        return "Squeeze" + str(self.dims)
