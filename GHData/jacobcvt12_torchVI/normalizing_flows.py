import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Bernoulli, Normal

# simulate data
n, d = 100000, 5
torch.manual_seed(1)
X = torch.randn(n, d)
X[:, 0] = 1.
beta_true = torch.randn(d)
y = Bernoulli(logits=X.mm(beta_true.view(d, 1))).sample() 


# define normalizing flows model
class NormalizingFlows(nn.Module):
    def __init__(self, D, K, h=torch.tanh, 
                 hprime=lambda x: 1 - torch.tanh(x).pow(2)):
        super(NormalizingFlows, self).__init__()
        self.D = D
        self.K = K
        self.h = h
        self.hprime = hprime
        self.w = nn.Parameter(torch.randn(D, K))
        self.u = nn.Parameter(torch.randn(D, K))
        self.b = nn.Parameter(torch.randn(K))
        self.mu = nn.Parameter(torch.randn(D))
        self.sigma = nn.Parameter(torch.randn(D))
        self.z_k = torch.empty(D, K + 1)

    def forward(self, x):
        z_0 = self.sample()
        self.transform(z_0)

        logits = x.mm(self.z_k[:, self.K].view(self.D, 1))

        return logits

    def sample(self):
        return self.mu + F.softplus(self.sigma) * torch.randn(self.D)

    def transform(self, z_0):
        self.z_k[:, 0] = z_0

        for k in range(self.K):
            weight = self.w[:, k].view(1, self.D)
            bias = self.b[k]
            neuron = self.h(weight.mm(self.z_k[:, k].view(self.D, 1)) + bias)
            self.z_k[:, k+1] = self.z_k[:, k] + self.u[:, k] * neuron


class ELBO(nn.Module):
    def __init__(self, model):
        super(ELBO, self).__init__()
        self.nf = model

    def forward(self, outputs, labels):
        loglik = Bernoulli(logits=outputs).log_prob(labels).sum()
        log_prior = Normal(0, 1).log_prob(self.nf.z_k).sum()
        log_joint = log_lik + log_prior

        mu, sigma = self.nf.mu, F.softplus(self.nf.sigma)
        K, zk = self.nf.K, self.nf.z_k
        w, b, u = self.nf.w, self.nf.b, self.nf.u
        hprime = self.nf.hprime


        entropy = Normal(mu, sigma).log_prob(zk[:, K+1]).sum()

model = NormalizingFlows(d, 2)
model(X)

mu, sigma = model.mu, F.softplus(model.sigma)
K, zk = model.K, model.z_k
w, b, u = model.w, model.b, model.u
hprime = model.hprime

# zk[:, 0:8].size()
# zk[:, 0]
# w[:, 0]
# w[:, 0].dot(zk[:, 0])
# w[:, 1].dot(zk[:, 1])
# w.t().mm(zk[:, 0:8]).diag()
(w * zk[:, 0:K]).sum(0)
(w * zk[:, 0:K]).sum(0) + b
hprime((w * zk[:, 0:K]).sum(0) + b)
hprime((w * zk[:, 0:K]).sum(0) + b)[0] * w[:, 0]
hprime((w * zk[:, 0:K]).sum(0) + b) * w
u[:, 0].view(d, 1).mm((hprime((w * zk[:, 0:K]).sum(0) + b)[0] * w[:, 0]).view(1, d))
torch.eye(d) + u[:, 0].view(d, 1).mm((hprime((w * zk[:, 0:K]).sum(0) + b)[0] * w[:, 0]).view(1, d))
(torch.eye(d) + u[:, 0].view(d, 1).mm((hprime((w * zk[:, 0:K]).sum(0) + b)[0] * w[:, 0]).view(1, d))).det()


