import json
import torch
from vae_skeleton import Conv2DVAE, Conv3dVAE
from torchsummary import summary
from typing import List


def get_2d_vae(layers: List, device: torch.device, latent_dim: int, x_dim: int,
               in_channels: int, json_data):
    return Conv2DVAE(x_dim=x_dim, channels=in_channels, layers=layers, latent_dim=latent_dim,
                     device=device, json_data=json_data)


def get_2d_beta_vae(layers: List, device: torch.device, latent_dim: int, beta: float, x_dim: int,
                    in_channels: int, json_data):
    return Conv2DVAE(x_dim=x_dim, channels=in_channels, layers=layers, latent_dim=latent_dim,
                     device=device, beta=beta, json_data=json_data)


def get_2d_gamma_vae(layers: List, device: torch.device, latent_dim: int, gamma: float, x_dim: int,
                     in_channels: int, json_data):
    return Conv2DVAE(x_dim=x_dim, channels=in_channels, layers=layers, latent_dim=latent_dim,
                     device=device, gamma=gamma, json_data=json_data)


def get_3d_vae(layers: List, device: torch.device, latent_dim: int, x_dim: int, z_dim: int,
               in_channels: int, json_data):
    return Conv3dVAE(x_dim=x_dim, z_dim=z_dim, channels=in_channels, layers=layers, latent_dim=latent_dim,
                     device=device, json_data=json_data)


def get_3d_beta_vae(layers: List, device: torch.device, latent_dim: int, beta: float, x_dim: int, z_dim: int,
                    in_channels: int, json_data):
    return Conv3dVAE(x_dim=x_dim, z_dim=z_dim, channels=in_channels, layers=layers, latent_dim=latent_dim,
                     device=device, beta=beta, json_data=json_data)


def get_3d_gamma_vae(layers: List, device: torch.device, latent_dim: int, gamma: float, x_dim: int, z_dim: int,
                     in_channels: int, json_data):
    return Conv3dVAE(x_dim=x_dim, z_dim=z_dim, channels=in_channels, layers=layers, latent_dim=latent_dim,
                     device=device, gamma=gamma, json_data=json_data)


def get_model(json_config_filepath: str):
    with open(json_config_filepath) as json_file:
        data = json.load(json_file)
        three_d: bool = data['3D']
        layers: List = data['layers']
        device_str: str = data['device']
        latent_dim: int = data['latent_dim']
        beta_vae: bool = data['beta_vae']
        beta: float = data['beta']
        gamma_vae: bool = data['gamma_vae']
        gamma: float = data['gamma']
        x_dim: int = data['x_dim']
        z_dim: int = data['z_dim']
        in_channels: int = data['in_channels']

        device = torch.device(device_str)

        if three_d:
            if beta_vae:
                model = get_3d_beta_vae(layers=layers, device=device, latent_dim=latent_dim, beta=beta, x_dim=x_dim,
                                        z_dim=z_dim, in_channels=in_channels, json_data=data)
                summary(model.cpu(), (in_channels, z_dim, x_dim, x_dim), device="cpu")
                return model
            elif gamma_vae:
                model = get_3d_gamma_vae(layers=layers, device=device, latent_dim=latent_dim, gamma=gamma, x_dim=x_dim,
                                         z_dim=z_dim, in_channels=in_channels, json_data=data)
                summary(model.cpu(), (in_channels, z_dim, x_dim, x_dim), device="cpu")
                return model
            else:
                model = get_3d_vae(layers=layers, device=device, latent_dim=latent_dim, x_dim=x_dim, z_dim=z_dim,
                                   in_channels=in_channels, json_data=data)
                summary(model.cpu(), (in_channels, z_dim, x_dim, x_dim), device="cpu")
                return model
        else:
            if beta_vae:
                model = get_2d_beta_vae(layers=layers, device=device, latent_dim=latent_dim, beta=beta, x_dim=x_dim,
                                        in_channels=in_channels, json_data=data)
                summary(model.cpu(), (in_channels, z_dim, x_dim, x_dim), device="cpu")
                return model
            elif gamma_vae:
                model = get_2d_gamma_vae(layers=layers, device=device, latent_dim=latent_dim, gamma=gamma, x_dim=x_dim,
                                         in_channels=in_channels, json_data=data)
                summary(model.cpu(), (in_channels, z_dim, x_dim, x_dim), device="cpu")
                return model
            else:
                model = get_2d_vae(layers=layers, device=device, latent_dim=latent_dim, x_dim=x_dim,
                                   in_channels=in_channels, json_data=data)
                summary(model.cpu(), (in_channels, z_dim, x_dim, x_dim), device="cpu")
                return model
