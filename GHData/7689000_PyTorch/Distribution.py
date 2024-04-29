import torch
from torch import FloatTensor
from torch.autograd import Variable
import torchvision
import numpy as np

# about Bernoulli distribution
from torch.distributions.bernoulli import Bernoulli
dist = Bernoulli(torch.tensor([0.3,0.6,0.9]))
print(dist)
dist.sample() # sample is binary, it takes 1 with p and 0 with 1-p

from torch.distributions.beta import Beta
dist = Beta(torch.tensor([0.3]), torch.tensor([0.5]))
print(dist)
dist.sample()

from torch.distributions.binomial import Binomial
# count of trials: 100
# 0, 0.2, 0.8, and 1 are event probabilities.
dist = Binomial(100, torch.tensor([0, .2, .8, 1]))
print(dist)
dist.sample()

from torch.distributions.categorical import Categorical
# 0.2, 0.2, 0.2, 0.2, 0.2 are event probabilities.
dist = Categorical(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))
print(dist)
dist.sample()

from torch.distributions.laplace import Laplace
# Laplace distribution parameterized by ’loc’ and ’scale’
dist = Laplace(torch.tensor([10.0]), torch.tensor([0.990]))
print(dist)
dist.sample()

from torch.distributions.normal import Normal
# Normal distribution parameterized by ’loc’ and ’scale’
dist = Normal(torch.tensor([100.0]), torch.tensor([10.0]))
print(dist)
dist.sample()
