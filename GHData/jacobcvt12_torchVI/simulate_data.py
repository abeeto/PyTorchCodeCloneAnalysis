import torch
from torch.distributions.bernoulli import Bernoulli

# logistic regression data
def logistic_regr(n, d):
    X = torch.randn(n, d)
    X[:, 0] = 1.
    beta_true = torch.randn(d)
    y = Bernoulli(logits=X.mm(beta_true.view(d, 1))).sample() 

    return X, y, beta_true

