import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import autograd
from collections import namedtuple
import math
import operator
from functools import reduce


class Model(nn.Module):
    r"""A NN mapping screen observations to discrete actions, as well as a valuation function

    Args:
        ob_space (tuple): shape of input, of the format `(width, height, channels)`
        ac_space (int): number of possible actions
        timestep_limit (int): maximum number of previous frames to backpropogate through

    Inputs: input, state_in
        - **input** (width, height, channels): tensor containing raw input pixels
        - **state_in** (State): internal state, either from `get_initial_state()` or `state_out` from a previous call

    Outputs: policy, vf, state_out
        - **policy** (ac_space): log probabilities of taking each action
        - **vf** (scalar): valuation of the current state
        - **state_out** (State): internal state, to be fed into next call
    """
    def __init__(self, ob_space, ac_space, is_cuda=False):
        super(Model, self).__init__()
        self.ob_space = list(ob_space)
        self.ac_space = ac_space
        self.is_cuda = is_cuda
        num_conv_layers = 4
        conv_channels = 32
        channels = [self.ob_space[-1]] + [conv_channels]*num_conv_layers

        self.conv = [nn.Conv2d(i, o, kernel_size=3, stride=2, padding=1)
                     for (i, o) in zip(channels, channels[1:])]

        for i, c in enumerate(self.conv):
            self.add_module('conv_{}'.format(i), c)

        self.conv_size_out = tuple([math.ceil(d / 2**num_conv_layers) for d in ob_space[0:2]] + [channels[-1]])
        self.lstm_size_in = reduce(operator.mul, self.conv_size_out)
        self.lstm_size_state = 256
        self.lstm = nn.LSTMCell(input_size=self.lstm_size_in,
                                hidden_size=self.lstm_size_state)
        self.logits = nn.Linear(self.lstm_size_state, ac_space)
        self.vf = nn.Linear(self.lstm_size_state, 1)

        for p in self.parameters():
            if p.ndimension() >= 2:  # xavier initialization doesn't apply to biases
                nn_init.xavier_uniform(p.data)
            else:
                p.data.zero_()  # initialize biases to zero
        # self.logits.weight.data = normalized_columns_initializer(self.logits.weight.data, 1.0)
        # self.vf.weight.data = normalized_columns_initializer(self.vf.weight.data, 1.0)
        self.share_memory()
        if self.is_cuda:
            self.cuda()
        self.train()

    def forward(self, inputs: autograd.Variable, state_in: 'ModelState') -> \
            (autograd.Variable, autograd.Variable, 'ModelState'):
        r"""Run the network to get

        Args:
            inputs: observation from environment
            state_in: state of LSTM cell, either from `get_initial_state` or a previous call to `forward`.

        Returns:
            (policy, vf, state_out)
            * policy (Variable): policy for choosing an action. Size: `(self.ac_space,)`,
                with entry `a` containing log probability choosing action `a`.
            * vf (Variable): estimation of the value of the current state (before taking any action). Size: `
            * state_out (State): LSTM cell state. Should be fed back into the next call to `forward`.
        """
        x = inputs  # shape: (channels, height, width)
        x = x.unsqueeze(0)  # shape: (1, channels, height, width)
        for i in range(4):
            x = self.conv[i](x)
            x = F.elu(x)
        # x shape: (1, channels_final, height_final, width_final)

        x = x.view(1, self.lstm_size_in)
        # x shape: (1, self.lstm_size_in)

        x, c = self.lstm(x, state_in)
        state_out = ModelState(x, c)
        # x shape: (1, self.lstm_size_state)

        logits = self.logits(x)
        policy = F.log_softmax(logits).squeeze(0)  # shape: (ac_space)

        vf = self.vf(x).squeeze()  # shape: (1)
        return policy, vf, state_out

    def get_initial_state(self) -> 'ModelState':
        r"""Returns the initial state of the LTSM cell.

        The state should reset to this at the beginning of each episode.

        """
        z = torch.zeros(1, self.lstm_size_state)
        h = autograd.Variable(z)
        c = autograd.Variable(z)
        if self.is_cuda:
            h = h.cuda()
            c = c.cuda()
        return ModelState(h, c)

    def grad_norm(self) -> float:
        r"""Returns the L2 norm of the gradient.

        Used for diagnostics

        """
        return math.sqrt(sum(torch.sum(p.grad.data**2) for p in self.parameters()))

    def param_norm(self) -> float:
        r"""Returns the L2 norm of the model parameters.

        Used for diagnostics.

        """
        return math.sqrt(sum(torch.sum(p.data**2) for p in self.parameters()))

    def clone(self) -> 'Model':
        r"""Returns a deep copy of this model.

        This is used to create local copies of the model for each worker.

        """
        new_model = Model(self.ob_space, self.ac_space, self.is_cuda)
        for param_self, param_other in self._param_pairs(new_model):
            param_other.data = param_self.data.clone()
        return new_model

    def load_gradients(self, other: 'Model'):
        r"""Load the gradients from another model into this model

        Args:
            other: The model which gradients are loaded FROM.

        """
        for param_self, param_other in self._param_pairs(other):
            if param_self.grad is None:
                param_self.sum().backward()  # hack to make `param_self.grad` exist, so we can copy into it
            param_self.grad.data.copy_(param_other.grad.data)

    def load_parameters(self, other: 'Model'):
        r"""Load parameters from another model into this model.

        Args:
            other: The model which parameters are loaded FROM

        """
        for param_self, param_other in self._param_pairs(other):
            param_self.data.copy_(param_other.data)

    def _param_pairs(self, other: 'Model'):
        r"""Get the pairs of :class:`nn.Parameter` objects with matching names.

        Args:
            other: The model whose parameters should be matched with `self`'s parameters

        Returns:
            Generator[(nn.Parameter, nn.Parameter)]: where the first parameter in each pair is from `self`,
                and the second is from `other`.
        """
        param_pairs = zip(self.named_parameters(), other.named_parameters())
        for (name_self, param_self), (name_other, param_other) in param_pairs:
            assert(name_self == name_other)
            yield (param_self, param_other)

class ModelState(namedtuple("ModelState", ["hidden", "cell"])):
    r"""Holds the state of the LSTM cell in a model

    Args:
        hidden: the hidden/output state
        cell: the cell state

    """
    def detach_(self):
        self.hidden.detach_()
        self.cell.detach_()
