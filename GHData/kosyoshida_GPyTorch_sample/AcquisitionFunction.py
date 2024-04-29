import torch
import torch.nn.functional as F

class ExpectedImprovement(object):
    def __init__(self):
        self.m = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

    def get(self, mean_y, covar_y, y_min):
        z = (y_min.expand_as(mean_y) - mean_y)/covar_y
        z = F.relu(z)
        ei = covar_y*(z*self.m.cdf(z) + torch.exp(self.m.log_prob(z)))
        return ei
