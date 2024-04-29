import yaml
import argparse

from models import *
from plot_latent_space import plot_latent_space, plot_all_latent_spaces, \
    plot_all_latent_vectors, plot_random_latent_spaces


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

model = vae_models[config['model_params']['name']](**config['model_params'])

model_dict = torch.load('checkpoints/VanillaVAE (CelebA, LD=40, LR=0.001).ckpt', map_location=torch.device('cpu'))
newdict = dict()
for k, v in model_dict['state_dict'].items():
    knew = k.split('.', 1)[1:][0]
    newdict[knew] = v

model.load_state_dict(newdict)
model.eval()

plot_latent_space(model, 16, 15, 5)
plot_all_latent_vectors(model)
plot_all_latent_spaces(model)
plot_random_latent_spaces(model, 100)
