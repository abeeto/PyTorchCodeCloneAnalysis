import torch
import torch.optim as optim

import os
import argparse
import shutil

from generators.generator import *
from transfer_network import Net
from data_loader import get_loader
from utils import save_result, get_config, prepare_sub_folder
from losses import VGG19, perceptual_loss, l1_loss

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2shoes.yaml', help='Path to the config file.')
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)

for key in config.keys():
    print("{} : {}".format(key, config[key]))
 
num_epochs    = config['num_epochs']
num_train     = config['num_train']
num_save      = config['display_size']
img_size      = config['img_size']
   
img_save_it   = config['image_save_iter']
model_save_it = config['snapshot_save_iter']

batch_size    = config['batch_size']
z_dim         = config['z_dim']
   
lr_z          = config['lr_z']
lr_t          = config['lr_t']
   
step_size     = config['step_size']
gamma         = config['gamma']
   
l1_w          = config['l1_w']
vgg_w         = config['vgg_w']
   
gen_ckpt      = config['gen_ckpt']
   
vgg_ckpt      = config['vgg_ckpt']

dataset       = config['src_dataset_train']

model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join("outputs", model_name)
checkpoint_directory, image_directory, _ = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

def main():
    # Get data loader
    loader = get_loader(config)

    # Latent Z for training set
    z_y = torch.randn(num_train, z_dim).cuda()
    z_y = z_y.view(num_train, z_dim, 1, 1)
    z_y.requires_grad = True

    optimizer_Z = optim.Adam([z_y], lr=lr_z)
    scheduler_Z = optim.lr_scheduler.StepLR(optimizer_Z, step_size=step_size, gamma=gamma)

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
    T.weight_init(mean=0.0, std=0.02)
    
    optimizer_T = optim.Adam(T.parameters(), lr=lr_t)
    scheduler_T = optim.lr_scheduler.StepLR(optimizer_T, step_size=step_size, gamma=gamma)

    vgg19 = VGG19(vgg_ckpt).cuda()

    for epoch in range(num_epochs):
        for step, data in enumerate(loader):
            if dataset in ['svhn', 'mnist']:
                data = data[0]
            source = data.cuda()

            z = z_y[step*batch_size:(step+1)*batch_size]
            
            T.eval()
            # Update Z vector
            target = G(z)
            target_downsampled = T.get_downsampled_images(target)
            target2source = T(target_downsampled)
                
            source_features = vgg19(source)
            target2source_features = vgg19(target2source)
            
            l1_loss_samples_z = l1_loss(target2source, source)
            perceptual_loss_samples_z = perceptual_loss(target2source_features, source_features)
            
            loss_z = l1_w * l1_loss_samples_z + vgg_w * perceptual_loss_samples_z
            loss_z = loss_z.mean()

            optimizer_Z.zero_grad()
            loss_z.backward()
            optimizer_Z.step()

            T.train()
            # Update the T network
            target = G(z)
            target_downsampled = T.get_downsampled_images(target)
            target2source = T(target_downsampled)
            
            source_features = vgg19(source)
            target2source_features = vgg19(target2source)
            
            l1_loss_samples_t = l1_loss(target2source, source)
            perceptual_loss_samples_t = perceptual_loss(target2source_features, source_features)
            
            loss_t = l1_w * l1_loss_samples_t + vgg_w * perceptual_loss_samples_t
            loss_t = loss_t.mean()

            optimizer_T.zero_grad()
            loss_t.backward()
            optimizer_T.step()

            print("Epoch: {}, Step: {}, Loss: {}, L1: {}, VGG: {}".format(epoch, step, loss_t, l1_loss_samples_t.mean(), perceptual_loss_samples_t.mean()))
                
        scheduler_Z.step()
        scheduler_T.step()

        if (epoch + 1) % img_save_it == 0:
            save_result(source, target, epoch, num_save, image_directory)

        if (epoch + 1) % model_save_it == 0:
            torch.save(T.state_dict(), os.path.join(checkpoint_directory, "transfer_network_param_{}.pkl".format(epoch)))

    torch.save(T.state_dict(), os.path.join(checkpoint_directory, "transfer_network_param_final.pkl"))
     
if __name__ == '__main__':
    main()