import torch
import math

import numpy as np
import torch.nn as nn

from torch.autograd import Variable

class BinarizingAutoencoder(nn.Module):
    def __init__(self, args):
        super(BinarizingAutoencoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.lambda_reg = args.lambda_reg

        assert math.pow(2, self.hidden_dim) > self.vocab_size, \
            "Hidden dimension is too small to encoder the vocabulary!"

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding.weight.requires_grad = False

        if type(args.embedding) == np.array:
            self.embedding.weight.data = args.embedding
        else:
            raise ValueError('Embedding weights is not numpy.array!')

        self.encoder = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        self.decoder = nn.Linear(self.hidden_dim, self.embed_dim)

        """
        Weight tying used for autoencoder as in the paper
        """
        if args.weight_tying:
            self.encoder.weight = self.decoder.weight

        self.init_params()

    def init_params(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    @classmethod
    def heaviside(cls, input):
        """
        :param input: a torch.FloatTensor or a torch.cuda.FloatTensor
        :return: the Heaviside function of the input

        >>> model = BinarizingAutoencoder(args)
        >>> input = torch.FloatTensor([1.0, -1.0])
        >>> print(model.heaviside(input))
        1
        0
        [torch.FloatTensor of size 2]
        """
        return (input >= 0).float()

    @classmethod
    def eye_like(cls, input):
        """
        :param input: a torch.FloatTensor or a torch.cuda.FloatTensor of size [m] or [m x m]
        :return: a Variable of unit Tensor of dimension [m x m]

        >>> model = BinarizingAutoencoder(args)
        >>> input = torch.FloatTensor([1.0, -1.0])
        >>> print(model.eye_like(input))
        1  0
        0  1
        [torch.FloatTensor of size 2x2]
        """
        if len(input.size()) == 2:
            assert input.size()[0] == input.size()[1], "The input size is not a square matrix"
        elif len(input.size()) > 2:
            raise ValueError("Input size should be dimension 2 or 1")
        return Variable(torch.eye(input.size()[0]))

    def forward(self, input):
        in_embed = self.embedding(input)
        binary = self.heaviside(self.encoder(in_embed))

        out_embed = self.decoder(binary)
        correlation = torch.mm(self.decoder.weight, self.encoder.weight)
        reg_loss = torch.norm(correlation - self.eye_like(correlation))
        return in_embed, out_embed, reg_loss