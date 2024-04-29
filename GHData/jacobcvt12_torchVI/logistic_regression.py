import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

# constants
n, d = 100000, 100
prior = Normal(0, 1) # prior on logistic regression parameters
EPOCHS = 200 

# generate data
from simulate_data import logistic_regr
torch.manual_seed(1)
X, y, beta_true = logistic_regr(n, d)

# function to sample from q using reparam method
def sample_q_normal(Lambda):
    mu = Lambda[:, 0]
    sigma = F.softplus(Lambda[:, 1])

    epsilon = torch.randn((mu.size()))
    z = mu + epsilon * sigma

    return z, mu, sigma


class LogisticRegression(nn.Module):
    """Bayesian Logistic Regression with VI"""
    def __init__(self, d):
        super(LogisticRegression, self).__init__()
        self.d = d
        self.Lambda = nn.Parameter(torch.randn((d, 2)))
        self.mu = self.Lambda[:, 0]
        self.sigma = F.softplus(self.Lambda[:, 1])
        self.z = self.Lambda[:, 0]

    def forward(self, input):
        self.truncate()
        self.mu = self.Lambda[:, 0]
        self.sigma = F.softplus(self.Lambda[:, 1])

        epsilon = torch.randn((self.mu.size()))
        self.z = self.mu + epsilon * self.sigma

        logits = input.mm(self.z.view(self.d, 1))

        return logits

    def truncate(self):
        with torch.no_grad():
            small = self.Lambda[:, 1] < torch.tensor(-11.5)
            self.Lambda[small, 1] = torch.tensor(-11.5)

    def log_lik(self, yhat, y):
        return Bernoulli(logits=yhat).log_prob(y).sum()

    def log_prior(self):
        return prior.log_prob(self.z).sum()

    def entropy(self):
        return Normal(self.mu, self.sigma).log_prob(self.z).sum()


class Elbo(nn.Module):
    def __init__(self, model):
        super(Elbo, self).__init__()
        self.model = model

    def forward(self, yhat, y):
        return model.entropy() - \
               model.log_lik(yhat, y) - \
               model.log_prior()


model = LogisticRegression(d)
criterion = Elbo(model)
opt = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for epoch in range(EPOCHS):
    opt.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        print(epoch, loss.item())

model.mu - beta_true
