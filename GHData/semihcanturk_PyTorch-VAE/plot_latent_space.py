import yaml
import argparse
import os

from models import *
import torchvision.utils as vutils


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='./configs/vae_vis.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

def plot_latent_space(model, dim1=None, dim2=None, sample_dim=10, dir='outputs/latent_space/'):
    if dim1 is None:
        dim1 = np.random.randint(0, config['model_params']['latent_dim'])
    if dim2 is None:
        dim2 = np.random.randint(0, config['model_params']['latent_dim'])

    grid = model.plot_latent_space(dim1, dim2, sample_dim)
    dir = dir + f"{config['exp_params']['dataset']}_{config['model_params']['name']}_LD={config['model_params']['latent_dim']}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    vutils.save_image(grid, dir + f"latent_space_{dim1},{dim2}.png", normalize=True, nrow=sample_dim)

def plot_all_latent_spaces(model, sample_dim=10, dir='outputs/latent_space/'):
    for i in range(config['model_params']['latent_dim']):
        for j in range(config['model_params']['latent_dim']):
            if i < j:
                plot_latent_space(model, i, j, sample_dim, dir)

def plot_all_latent_vectors(model, sample_dim=5, dir='outputs/latent_space/vectors/'):
    for i in range(config['model_params']['latent_dim']):
        plot_latent_space(model, i, i, sample_dim, dir)

def plot_random_latent_spaces(model, num_samples=20):
    for i in range(num_samples):
        plot_latent_space(model)
