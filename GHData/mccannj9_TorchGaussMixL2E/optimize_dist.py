#! /usr/bin/env python3

import numpy as np
np.random.seed(3)

import torch
from torch.distributions import MultivariateNormal as MVN
torch.manual_seed(1000)

softplus = torch.nn.Softplus(beta=1)
softmax = torch.nn.Softmax(dim=0)

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

# data to fit a distribution
# will expand to GMM later
data = X[X[:,2] == 0, 0:2]
print(data.shape)

# torch initialize to fit distribution
mu_opt = torch.randn(D, requires_grad=True)
tril_entries = torch.randn(D*(D+1)//2, requires_grad=True)

def update_tril(entries, D):
    tril = torch.zeros(D, D)
    tril[range(D), range(D)] = softplus(entries[0:D])
    off_idx = torch.tril_indices(D, D)[0] != torch.tril_indices(D, D)[1]
    a, b = torch.tril_indices(D, D)[:, off_idx]
    tril[a, b] = entries[D:]
    return tril

mvn = MVN(loc=mu_opt, scale_tril=update_tril(tril_entries, D))

opt = torch.optim.Adam([mu_opt, tril_entries], lr=0.01)
print(opt.param_groups)
for i in range(10):
    opt.zero_grad()
    loss = (-mvn.log_prob(torch.Tensor(data))).sum()
    loss.backward(retain_graph=True)
    opt.step()
    if i % 100 == 0:
        print(f"{i}: {loss.item()}")
        print(mvn.scale_tril @ mvn.scale_tril.T)
        print(tril_entries)
    mvn = MVN(mu_opt, scale_tril=update_tril(tril_entries, D))

pi = torch.randn(K, requires_grad=True)
mu_opt = torch.randn(K, D, requires_grad=True)
tril_entries = torch.randn(K, D*(D+1)//2, requires_grad=True)
opt = torch.optim.Adam([pi, mu_opt, tril_entries], lr=0.1)

def update_mvns(pi, mu_opt, tril_entries):
    mvns = [
        MVN(
            loc=mu_opt[i], scale_tril=update_tril(tril_entries[i], D), validate_args=True
        ) for i in range(K)
    ]

    diffs = []
    for i in range(K):
        for j in range(K):
            m = MVN(
                mu_opt[i] - mu_opt[j],
                covariance_matrix=mvns[i].covariance_matrix + mvns[j].covariance_matrix,
                validate_args=True
            )
            diffs.append((m, pi[i], pi[j]))
    return mvns, diffs

print("TRIL", tril_entries[0], tril_entries[0].shape)
mvns, diffs = update_mvns(pi, mu_opt, tril_entries)
inputs = torch.Tensor(X[:, 0:2])
for i in range(1000):
    opt.zero_grad()
    prob_losses = 0
    for j in range(K):
        prob_losses += (pi[j] * mvns[j].log_prob(inputs).exp()).sum()
    prob_losses *= (-2/inputs.shape[0])
    l2_losses = 0
    for d, w1, w2 in diffs:
        l2_losses += d.log_prob(torch.zeros(D)).exp() * w1 * w2
    loss = prob_losses + l2_losses
    if i % 100 == 0:
        print(f"{i}: {loss.item()}")
    loss.backward(retain_graph=True)
    opt.step()
    mvns, diffs = update_mvns(pi, mu_opt, tril_entries)
