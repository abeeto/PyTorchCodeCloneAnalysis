import torch
import torch.optim as optim

import os
import argparse
import shutil
import sys

from generators.generator import *
from transfer_network import Net
from data_loader import get_loader
from utils import save_result, get_config, prepare_sub_folder, get_model_list
from losses import VGG19, perceptual_loss, l1_loss

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2shoes.yaml', help='Path to the config file.')
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)

for key in config.keys():
    print("{} : {}".format(key, config[key]))

img_size      = config['img_size']
num_save      = config['display_size']
z_dim         = config['z_dim']
   
lr_z          = config['lr_z']
   
l1_w          = config['l1_w']
vgg_w         = config['vgg_w']
   
gen_ckpt      = config['gen_ckpt']
   
vgg_ckpt      = config['vgg_ckpt']

num_sample    = config['num_sample']
num_eval_iter = config['num_eval_iter']

dataset       = config['src_dataset_train']

model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join("outputs", model_name)
checkpoint_directory, _, evaluation_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))


def main():
    # Get data loader
    loader = get_loader(config, train=False)

    num_test = len(loader.dataset)

    # Generator network for target domain
    if img_size == 64:
        G = Generator64().cuda()
    else:
        G = Generator32().cuda()
    g_ckpt = torch.load('models/' + gen_ckpt)
    G.load_state_dict(g_ckpt)
    G.eval()

    # Transfer network from target to source domain
    T = Net(config).cuda()
    last_model_name = get_model_list(checkpoint_directory)
    t_ckpt = torch.load(last_model_name)
    T.load_state_dict(t_ckpt)
    T.eval()

    vgg19 = VGG19(vgg_ckpt).cuda()

    for idx in range(num_test):

        # Source image
        source = loader.dataset[idx]
        if dataset in ['svhn', 'mnist']:
            source = source[0]
        source = source.cuda()
        source = source.unsqueeze(0)
        source = source.repeat(num_sample, 1, 1, 1)
        
        # Latent Z for training set
        z = torch.randn(num_sample, z_dim, 1, 1).cuda()
        z.requires_grad = True

        optimizer_Z = optim.Adam([z], lr=lr_z)
        
        for it in range(num_eval_iter):            
            # Update Z vector
            target = G(z)
            target_downsampled = T.get_downsampled_images(target)
            target2source = T(target_downsampled)
                
            source_features = vgg19(source)
            target2source_features = vgg19(target2source)
            
            l1_loss_samples_z = l1_loss(target2source, source)
            perceptual_loss_samples_z = perceptual_loss(target2source_features, source_features)
            
            loss_z = l1_w * l1_loss_samples_z + vgg_w * perceptual_loss_samples_z
            loss_z_samples = loss_z
            loss_z = loss_z.mean()

            optimizer_Z.zero_grad()
            loss_z.backward()
            optimizer_Z.step()

            print("Image: {}, Step: {}, Loss: {}, L1: {}, VGG: {}".format(idx, it, loss_z, l1_loss_samples_z.mean(), perceptual_loss_samples_z.mean()))


        save_result(source, target, idx, num_save, evaluation_directory, loss_z_samples)
     
if __name__ == '__main__':
    main()