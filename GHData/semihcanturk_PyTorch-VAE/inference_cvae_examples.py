import yaml
import argparse

from models import *
from inference_cvae import plot_cvae_classes, plot_cvae_vector


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='./configs/cvae_vis.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])

model_dict = torch.load('checkpoints/CelebA-CVAE-LD=40-LR=0.0001.ckpt', map_location=torch.device('cpu'))
newdict = dict()
for k, v in model_dict['state_dict'].items():
    knew = k.split('.', 1)[1:][0]
    newdict[knew] = v

model.load_state_dict(newdict)
model.eval()

plot_cvae_classes(model)
plot_cvae_vector(model, [0, 4, 7, 13, 14, 15, 16, 31])
plot_cvae_vector(model, [5, 21, 23, 25])
plot_cvae_vector(model, [2, 17, 20])
plot_cvae_vector(model, [2, 6, 8, 27, 29, 31, 39])