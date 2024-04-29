import torch
import torch.nn as nn
from models import DeterministicEncoder, LatentEncoder, Decoder


class NeuralProcess(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, z_dim: int, r_dim: int, s_dim: int, width: int = 200):
        super(NeuralProcess, self).__init__()
        self.determinstic_encoder = DeterministicEncoder(x_dim, y_dim, r_dim)
        self.latent_encoder = LatentEncoder(x_dim, y_dim, z_dim)
        self.decoder = Decoder(x_dim, z_dim, r_dim, y_dim)

    def get_mu_sigma(self, x, y):
        return self.latent_encoder(x, y)

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor, x_target: torch.Tensor):
        r = self.determinstic_encoder(x_context, y_context)
        z_mu, z_sigma = self.latent_encoder(x_context, y_context)
        z = torch.randn(z_mu.size()) * z_sigma + z_mu
        num_target_points = x_target.shape[0]
        y = self.decoder(torch.cat((x_target.reshape(-1,1), z.repeat(num_target_points, 1), r.repeat(num_target_points, 1)), dim=1))
        return y


class NeuralProcessLoss(nn.Module):
    def __init__(self):
        super(NeuralProcessLoss, self).__init__()
        self.mse_loss = 0
        self.kl_div = 0

    def forward(self, neural_process: NeuralProcess, x_context: torch.Tensor, y_context: torch.Tensor, x_target: torch.Tensor, y_target: torch.Tensor):
        mu_c, loc_c = neural_process.get_mu_sigma(x_context, y_context)
        mu_t, loc_t = neural_process.get_mu_sigma(x_target, y_target)
        dist_context = torch.distributions.Normal(mu_c, loc_c)
        dist_target = torch.distributions.Normal(mu_t, loc_t)
        d_kl = nn.KLDivLoss(dist_context, dist_target)
        y_pred = neural_process(x_context, y_context, x_target)
        ll_fit = torch.nn.functional.mse_loss(y_target, y_pred)
        self.kl_div = d_kl
        self.mse_loss = ll_fit
        return ll_fit + d_kl