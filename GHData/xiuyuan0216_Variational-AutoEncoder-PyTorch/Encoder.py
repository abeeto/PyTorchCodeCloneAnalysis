import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, D_in, H, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, H)
        self.enc_mu = nn.Linear(H, latent_size)
        self.enc_log_sigma = nn.Linear(H, latent_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return torch.distributions.Normal(mu, sigma)