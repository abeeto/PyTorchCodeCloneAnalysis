from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


import my_train_fns
from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from vca_utils import load_checkpoint

from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample)

import neptune.new as neptune
from neptune.new.types import File
import Constants

def run(config):

    neptune_run = neptune.init(project='bjohnbraddock/BigGAN-VCA-3', api_token = Constants.NEPTUNE_API_KEY, source_files=['*.py'])

    assert torch.cuda.is_available(), 'Torch could not find CUDA enabled GPU'
    device = 'cuda'

    # Seed RNG
    seed= config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    print("Loading BigGAN Generator")
    G = BigGAN.from_pretrained('biggan-deep-256', cache_dir=config['biggan_cache_dir']).to(device)

    print("Loading VCA")
    VCA = Amy_IntermediateRoad( lowfea_VGGlayer=10, highfea_VGGlayer=36, is_highroad_only=False, is_gist=False)
    VCA = load_checkpoint(VCA, config['vca_filepath'])
    VCA = VCA.to(device)
           

    # print(G)
    print('Number of params in G: {} '.format(sum([p.data.nelement() for p in G.parameters()])))


    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {
        'itr': 0,
        'epoch': 0,
        'config': config
    }

    state_dict['config'] = config
    

    neptune_run['config/model'] = 'BigGAN'
    neptune_run['config/criterion'] = 'VCA'
    neptune_run['config/optimizer'] = 'Adam'
    neptune_run['config/params'] = config

    
    batch_size = config['batch_size']

    # z_ = nn.init.trunc_normal_(torch.randn(G_batch_size, G.dim_z), mean=0, std=1, a=-2, b=2).to(device, torch.float32).requires_grad_()
    # y_ = nn.init.trunc_normal_(torch.randn(G_batch_size, G.shared_dim), mean=0, std=1, a=-2, b=2).to(device, torch.float32).requires_grad_()
    # latent_vector = nn.init.trunc_normal_(torch.randn(batch_size, config['dim_z']), mean=0, std=1, a=-2, b=2).to(device, torch.float32).requires_grad_()

    latent_vector = truncated_noise_sample(truncation=config['truncation'], batch_size=batch_size)
    latent_vector = torch.from_numpy(latent_vector).to(device).requires_grad_()

    class_vector = torch.mul(0.05, F.softmax(nn.init.trunc_normal_(torch.randn(batch_size, config['num_classes']), mean=0.5, a=0, b=1), dim=1)).to(device).requires_grad_()

    
    if config['optimize_latent']:
        z_y_optim = torch.optim.Adam([latent_vector, class_vector], lr=config['lr'])
    else:
        z_y_optim = torch.optim.Adam([class_vector], lr=config['lr'])


    train = my_train_fns.VCA_latent_training_function_alt(G, VCA, latent_vector, class_vector, z_y_optim, config)

    print(state_dict)


    with torch.no_grad():
        # Initial image log
        G.eval()
        G_z = torch.tensor(G(latent_vector, class_vector, config['truncation']))
        G_z = F.interpolate(G_z, size=224)
        VCA_G_z = VCA(G_z).view(-1)
        Gz_grid = torchvision.utils.make_grid(G_z.float(), normalize=True)
        Gz_grid = torch.permute(Gz_grid, (1,2,0))

        neptune_run['initial/latent_vector'] = latent_vector
        neptune_run['initial/class_vector'] = (class_vector)
        neptune_run['initial/class_max'] = (torch.topk(class_vector,5))
        neptune_run['initial/G_z'].upload(File.as_image(Gz_grid.cpu()))
        neptune_run['initial/vca_tensor'] = (VCA_G_z)

        neptune_run['train/latent_vector'].log(latent_vector)
        neptune_run['train/class_vector'].log(class_vector)
        neptune_run['train/class_max'].log(torch.topk(class_vector,5))
        neptune_run['train/torch_tensor'].log(File.as_image(Gz_grid.cpu()))
        neptune_run['train/vca_tensor'].log(VCA_G_z)
        
    

    for epoch in range(state_dict['epoch'], config['num_epochs']):
        print(f"Epoch: {epoch}")
        neptune_run['train/current_epoch']=epoch
        for i in range(config['iters_per_epoch']):
            state_dict['itr'] += 1

            G.eval()
            VCA.eval()
            metrics = train()

            neptune_run["training/batch/loss"].log(metrics['G_loss'])
            neptune_run["training/batch/acc"].log(metrics['VCA_G_z'])
            if not(state_dict['itr'] % config['log_every']):
                print('Epoch: {}    Itr: {}    G_loss: {:.4e}    VCA_G_z: {}'.format(state_dict['epoch'], state_dict['itr'], metrics['G_loss'], metrics['VCA_G_z']))
        print('Epoch: {}    Itr: {}    G_loss: {:.4e}    VCA_G_z: {}'.format(state_dict['epoch'], state_dict['itr'], metrics['G_loss'], metrics['VCA_G_z']))
       
        
        with torch.no_grad():
        # Epoch image log
            G.eval()
            G_z = torch.tensor(G(latent_vector, class_vector, config['truncation']))
            G_z = F.interpolate(G_z, size=224)
            VCA_G_z = VCA(G_z).view(-1)
            Gz_grid = torchvision.utils.make_grid(G_z.float(), normalize=True)
            Gz_grid = torch.permute(Gz_grid, (1,2,0))

            neptune_run['train/latent_vector'].log(latent_vector)
            neptune_run['train/class_vector'].log(class_vector)
            neptune_run['train/class_max'].log(torch.topk(class_vector,5))
            neptune_run['train/torch_tensor'].log(File.as_image(Gz_grid.cpu()))
            neptune_run['train/vca_tensor'].log(VCA_G_z)

        
        state_dict['epoch'] += 1

    # TODO: Save

    with torch.no_grad():
        # Final image log
        G.eval()
        G_z = torch.tensor(G(latent_vector, class_vector, config['truncation']))
        G_z = F.interpolate(G_z, size=224)
        VCA_G_z = VCA(G_z).view(-1)
        Gz_grid = torchvision.utils.make_grid(G_z.float(), normalize=True)
        Gz_grid = torch.permute(Gz_grid, (1,2,0))

        neptune_run['final/latent_vector'] = latent_vector
        neptune_run['train/class_vector'].log(class_vector)
        neptune_run['train/class_max'].log(torch.topk(class_vector,5))
        neptune_run['final/torch_tensor'].upload(File.as_image(Gz_grid.cpu()))
        neptune_run['final/vca_tensor'] = VCA_G_z


def main():

    parser = ArgumentParser(description='Parser for simplified refactor')
    parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use; affects both initialization and '
        ' dataloading. (default: %(default)s)')
    parser.add_argument(
    '--vca_filepath', type=str, default='',
    help='Relative filepath to trained VCA model .pth (default: %(default)s)'
    )
    parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
    '--dim_z', type=int, default=128,
    help='Noise dimensionality: %(default)s)')
    parser.add_argument(
    '--num_classes', type=int, default=1000,
    help='Number of classes: %(default)s')
    parser.add_argument(
    '--lr', type=float, default=1e-3,
    help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
    '--truncation', type=float, default=0.4,
    help='BigGAN truncation parameter for sampling normal distr. (default: %(default)s)')
    parser.add_argument(
    '--num_epochs', type=int, default=30,
    help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
    '--iters_per_epoch', type=int, default=200,
    help='Batch iterations per epoch used in VCA Generator Training (default: %(default)s)')
    parser.add_argument(
    '--log_every', type=int, default=250,
    help='Log VCA finetune metrics every X iterations (default: %(default)s')
    parser.add_argument(
    '--biggan_cache_dir', type=str, default='/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained',
    help='Where to cache BigGAN from TFHUB (default: %(default)s)'
    )
    parser.add_argument(
    '--train_unpleasant', action='store_true', default=False,
    help='Set to optimize VCA response of 0 (unpleasant) (default: %(default)s)')
    parser.add_argument(
    '--optimize_latent', action='store_true', default=False,
    help='Optmize the latent z (in addition to class vector) (default %(default)s)'
    )

    
    config = vars(parser.parse_args())

    print(config)
    run(config)

if __name__ == '__main__':
    main()