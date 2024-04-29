#! /usr/bin/env python3

import torch
from torch.nn import Linear, Module
from torch.distributions import MultivariateNormal as MVN
from torch.utils.tensorboard import SummaryWriter

import numpy as np

sigmoid = torch.nn.Sigmoid()
softmax = torch.nn.Softmax(dim=0)
softplus = torch.nn.Softplus(beta=1)

def update_tril(entries, D):
    tril = torch.zeros(D, D)
    tril[range(D), range(D)] = softplus(entries[0:D])
    off_idx = torch.tril_indices(D, D)[0] != torch.tril_indices(D, D)[1]
    a, b = torch.tril_indices(D, D)[:, off_idx]
    tril[a, b] = entries[D:]
    return tril

def update_mvns(pi, mu_opt, tril_entries):
    mvns = [
        MVN(
            loc=mu_opt[i], scale_tril=update_tril(
                tril_entries[i], mu_opt.shape[-1]
            ), validate_args=True
        ) for i in range(pi.shape[0])
    ]

    diffs = []
    for i in range(pi.shape[0]):
        for j in range(pi.shape[0]):
            m = MVN(
                mu_opt[i] - mu_opt[j],
                covariance_matrix=mvns[i].covariance_matrix + mvns[j].covariance_matrix,
                validate_args=True
            )
            diffs.append((m, pi[i], pi[j]))
    return mvns, diffs


class GaussianMixtureModel(Module):
    def __init__(self, num_mixtures=2, num_dim=2, unity=True, full_cov=True, cuda_dev="cuda:0"):
        super().__init__()

        # model structure
        self.init = torch.nn.init.normal_
        self.k = num_mixtures
        self.d = num_dim
        self.full_cov = full_cov
        self.unity = unity

        if unity:
            self.weights_bijector = softmax
        else:
            self.weights_bijector = sigmoid

        # initialize parameters
        self.pi = self.init(torch.empty(self.k, requires_grad=True))

        self.mu = self.init(
            torch.empty(self.k, self.d, requires_grad=True)
        )

        if full_cov:
            self.tril_entries = self.init(
                torch.empty(
                    self.k, self.d * (self.d + 1) // 2, requires_grad=True
                )
            )

        else:
            self.tril_entries = self.init(
                torch.empty(self.k, self.d, requires_grad=True)
            )

        self.parameters_list = [self.pi, self.mu, self.tril_entries]
        self.mvns, self.diffs = update_mvns(*self.parameters_list)


        use_cuda = torch.cuda.is_available()
        self.device = torch.device(cuda_dev if use_cuda else "cpu")

        if use_cuda:
            self.cuda()

    def forward(self, inputs):
        self.pi = self.weights_bijector(self.pi)
        self.mvns, self.diffs = update_mvns(*self.parameters_list)

    def compute_probs_loss(self, inputs):
        loss = 0
        for x in range(self.k):
            loss += (self.pi[x] * self.mvns[x].log_prob(inputs).exp()).sum()
        return (-2/inputs.shape[0]) * loss

    def compute_l2_loss(self):
        l2_loss = 0
        for d, w1, w2 in self.diffs:
            l2_loss += d.log_prob(
                torch.zeros(self.d)
            ).exp() * w1 * w2
        return l2_loss


    def fit(self, inputs, nepochs=1000, starts=1, log_interval=10):

        optimizer = torch.optim.Adam
        opt = optimizer(self.parameters_list, lr=0.01)
        for x in range(starts):
            for i in range(nepochs):
                opt.zero_grad()
                loss = self.compute_probs_loss(inputs) + self.compute_l2_loss()
                loss.backward(retain_graph=True)
                opt.step()
                self.mvns, self.diffs = update_mvns(
                    *self.parameters_list
                )
                if i % log_interval == 0:
                    print(f"{x} {i}: {loss.item()}")

import sys

np.random.seed(int(sys.argv[1]))
torch.manual_seed(int(sys.argv[2]))


# simulate some data
N = 500
D = 2
K = 2

# true parameters of the model
weights = np.array([0.65, 0.35])
mus = np.array([[-1.0, -1.0], [1.0, 1.0]])
covs = np.array([
    [[0.11, -0.05], [-0.05, 0.11]],
    [[0.33, 0.100], [0.100, 0.33]]
])

X = np.zeros((N,D+1)) # accomodate labels in last col
for i in range(N):
    k = np.argmax(np.random.multinomial(1, weights))
    X[i, 0:2] = np.random.multivariate_normal(mus[k], covs[k])
    X[i, 2] = k

inputs = torch.Tensor(X[:, 0:2])

GMM = GaussianMixtureModel()
GMM.fit(inputs, nepochs=1000, log_interval=100)
