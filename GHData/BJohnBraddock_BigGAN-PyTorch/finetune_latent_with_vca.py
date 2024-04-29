import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


import utils
import train_fns
from sync_batchnorm import patch_replication_callback
from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from vca_utils import load_checkpoint
from converter import get_config

import neptune.new as neptune
from neptune.new.types import File
import Constants

def run(config):

    neptune_run = neptune.init(project='bjohnbraddock/BigGAN-VCA-2', api_token = Constants.NEPTUNE_API_KEY, source_files=['*.py'])

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]

    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)

    device = 'cuda'
    print("Using device cuda" if torch.cuda.is_available() else "Cuda device not found...")


    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))
    print('Experiment name is {}'.format(experiment_name))

    # Build the model
    p_config = get_config(256)
    G = model.Generator(**p_config).to(device)
    D = None
    G_ema, ema = None, None
    
    print("Loading VCA")
    VCA = Amy_IntermediateRoad( lowfea_VGGlayer=10, highfea_VGGlayer=36, is_highroad_only=False, is_gist=False)
    VCA = load_checkpoint(VCA, config['vca_filepath'])
    VCA = VCA.to(device)

    

    if config['G_fp16']:
        print('Casting G to float16')
        G = G.half()
           

    # print(G)
    print('Number of params in G: {} '.format(sum([p.data.nelement() for p in G.parameters()])))


    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {
        'itr': 0,
        'epoch': 0,
        'save_num': 0,
        'save_best_num': 0,
        'best_IS': 0,
        'best_FID': 999999,
        'config': config
    }


    if config['resume']:
        print('Loading weights')
        # utils.load_weights(G, D, state_dict,
        #                     config['load_weights_root'], experiment_name,
        #                     config['load_weights'] if config['load_weights'] else None,
        #                     G_ema if config['ema'] else None, load_optim=False)
        G_state_dict = torch.load(config['load_weights_root']+"/G.pth")
        G.load_state_dict(G_state_dict, strict=False)

    state_dict['config'] = config
    

    neptune_run['config/model'] = config['model']
    neptune_run['config/criterion'] = type(VCA).__name__
    neptune_run['config/optimizer'] = 'Adam'
    neptune_run['config/params'] = config

    
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    # z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'], truncated_z=config['truncated_z'], requires_grad=True)
    # z_.sample_()
    # y_.sample_()

    z_ = nn.init.trunc_normal_(torch.randn(G_batch_size, G.dim_z), mean=0, std=1, a=-2, b=2).to(device, torch.float32).requires_grad_()
    
    
    # The class category variable was initialized as αS(n) where α = 0.05, S is the softmax function, 
    # and n is a vector of scalars randomly sampled from the truncated normal distribution between [0, 1]
   

    # y_ = torch.zeros(G_batch_size).random_(0, config['n_classes']).to(device, torch.int64)
    y_ = torch.tensor([850]).to(device, torch.int64)
    # neptune_run['initial/y_'] = y_
    y_ = torch.tensor(G.shared(y_), device=device, requires_grad=True)
    
    # y_ = nn.init.trunc_normal_(torch.randn(G_batch_size, G.shared_dim), mean=0, std=1, a=-2, b=2).to(device, torch.float32).requires_grad_()
    print(z_.size())
    print(y_.size())
    
    z_y_optim = torch.optim.Adam([z_, y_], lr=config['G_lr'])


    train = train_fns.VCA_latent_training_function(G, VCA, z_, y_, z_y_optim, config)

    print(state_dict)


    with torch.no_grad():

            G.eval()
            
            fixed_Gz = torch.tensor(G(z_, y_))

            fixed_Gz = F.interpolate(fixed_Gz, size=224)

            VCA_G_z = VCA(fixed_Gz).view(-1)
            fixed_Gz_grid = torchvision.utils.make_grid(fixed_Gz.float(), normalize=True)
            fixed_Gz_grid = torch.permute(fixed_Gz_grid, (1,2,0))
            neptune_run['train/latent_vector'].log(z_)
            neptune_run['train/torch_tensor'].log(File.as_image(fixed_Gz_grid.cpu()))
            neptune_run['train/vca_tensor'].log(VCA_G_z)
            neptune_run['initial/latent_vector']= z_
            neptune_run['initial/G_z'].upload(File.as_image(fixed_Gz_grid.cpu()))
            neptune_run['initial/vca_tensor'] = (VCA_G_z)
            # neptune_run['initial/y_'] = y_
            image_filename = 'fixed_samples_{}.jpg'.format(1)
            torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)

    exit()
    

    # TODO: Training loop
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        print(f"Epoch: {epoch}")

        for i in range(config['iters_per_epoch']):
            state_dict['itr'] += 1

            G.eval()
            metrics = train()

            neptune_run["training/batch/loss"].log(metrics['G_loss'])
            neptune_run["training/batch/acc"].log(metrics['VCA_G_z'])
            if not(state_dict['itr'] % config['log_every']):
                print('Epoch: {}    Itr: {}    G_loss: {:.4e}    VCA_G_z: {}'.format(state_dict['epoch'], state_dict['itr'], metrics['G_loss'], metrics['VCA_G_z']))

        
        if config['G_eval_mode']:
            G.eval()
        print("Saving")
        print('Epoch: {}    Itr: {}    G_loss: {:.4e}    VCA_G_z: {}'.format(state_dict['epoch'], state_dict['itr'], metrics['G_loss'], metrics['VCA_G_z']))
       
        
        with torch.no_grad():
            
            fixed_Gz = torch.tensor(G(z_, y_))

            fixed_Gz = F.interpolate(fixed_Gz, size=224)

            VCA_G_z = VCA(fixed_Gz).view(-1)
            fixed_Gz_grid = torchvision.utils.make_grid(fixed_Gz.float(), normalize=True)
            fixed_Gz_grid = torch.permute(fixed_Gz_grid, (1,2,0))
            neptune_run['train/latent_vector'].log(z_)
            neptune_run['train/torch_tensor'].log(File.as_image(fixed_Gz_grid.cpu()))
            neptune_run['train/vca_tensor'].log(VCA_G_z)
            neptune_run['train/y_'].log(y_.cpu())

        
            
        
        state_dict['epoch'] += 1

    # TODO: Save
    # train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config, experiment_name)
    with torch.no_grad():
            G.eval()
            
            fixed_Gz = torch.tensor(G(z_, y_))

            fixed_Gz = F.interpolate(fixed_Gz, size=224)

            VCA_G_z = VCA(fixed_Gz).view(-1)

            fixed_Gz_grid = torchvision.utils.make_grid(fixed_Gz.float(), normalize=True)
            fixed_Gz_grid = torch.permute(fixed_Gz_grid, (1,2,0))
            neptune_run['final/latent_vector'] = z_
            neptune_run['final/torch_tensor'].upload(File.as_image(fixed_Gz_grid.cpu()))
            neptune_run['final/vca_tensor'] = VCA_G_z

            

def main():

    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    print(config)
    run(config)

if __name__ == '__main__':
    main()