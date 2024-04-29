import yaml
import argparse
import csv

from models import *
import torchvision.utils as vutils


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

with open('features_list.txt', mode='r') as infile:
    reader = csv.reader(infile)
    feature_dict = dict()
    i=0
    for row in reader:
        if len(row) > 0:
            feature_dict[i] = row[0].strip()
            i += 1

def plot_cvae_vector(model, idxs):
    input_vect = torch.zeros(config['exp_params']['batch_size'], config['model_params']['num_classes'])
    filename = ""
    for feature_idx in idxs:
        input_vect[:, feature_idx] = 1
        filename += feature_dict[feature_idx] + "_"
    out = model.infer(input_vect)
    vutils.save_image(out.cpu().data, f"outputs/"f"{filename}.png", normalize=True,
                      nrow=12)

def plot_cvae_classes(model):
    for feature_idx in range(config['model_params']['num_classes']):
        input_vect = torch.zeros(config['exp_params']['batch_size'], config['model_params']['num_classes'])
        input_vect[:, feature_idx] = 1
        out = model.infer(input_vect)
        vutils.save_image(out.cpu().data, f"outputs/"f"{feature_idx}_{feature_dict[feature_idx]}.png", normalize=True, nrow=12)




