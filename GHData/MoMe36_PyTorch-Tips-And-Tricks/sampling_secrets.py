import torch 
import torch.nn as nn 
from torch.distributions import Normal


def get_lp(dist, action): 

	return dist.log_prob(action).sum(-1, keepdim = True)


torch.manual_seed(1)

mean = torch.rand(1,3)
var = torch.exp(torch.zeros_like(mean))

print('Mean: {}\nVar: {}'.format(mean, var))
d = Normal(mean, var)

x = d.sample()
print('Sampling: {}'.format(x))


random_action = torch.zeros_like(x)
log_prob = d.log_prob(random_action)
other_lp = get_lp(d, random_action)

print('Random tensor: {} Log prob: {}'.format(random_action, log_prob))
print('Random tensor: {} Log prob 2: {}'.format(random_action, other_lp))

print('Probs: {} -- {}'.format(torch.exp(log_prob), torch.exp(other_lp)))
