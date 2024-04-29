import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


import utils
import train_fns
from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from vca_utils import load_checkpoint
from converter import get_config

import neptune.new as neptune
from neptune.new.types import File
import Constants

def run(config):

    neptune_run = neptune.init(project='bjohnbraddock/BigGAN-VCA', api_token = Constants.NEPTUNE_API_KEY, source_files=['*.py'])

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]

    
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
    p_config['no_optim'] = False
    p_config['G_lr'] = config['G_lr']
    G = model.Generator(**p_config).to(device)
    D = None
    G_ema, ema = None, None
    
    print("Loading VCA")
    VCA = Amy_IntermediateRoad( lowfea_VGGlayer=10, highfea_VGGlayer=36, is_highroad_only=False, is_gist=False)
    VCA = load_checkpoint(VCA, config['vca_filepath'])
    VCA = VCA.to(device)
           

    print(G)
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


    
    print('Loading weights')
    # utils.load_weights(G, D, state_dict,
    #                     config['load_weights_root'], experiment_name,
    #                     config['load_weights'] if config['load_weights'] else None,
    #                     G_ema if config['ema'] else None, load_optim=False)
    G_state_dict = torch.load(config['load_weights_root']+"/G.pth")
    G.load_state_dict(G_state_dict, strict=False)

    state_dict['config'] = config
    

    neptune_run['config/model'] = type(model).__name__
    neptune_run['config/criterion'] = type(VCA).__name__
    neptune_run['config/optimizer'] = type(G.optim).__name__
    neptune_run['config/params'] = config

    
    

    G_batch_size = config['batch_size']
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, p_config['n_classes'], device=device, fp16=p_config['G_fp16'])

    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z, p_config['n_classes'], device=device, fp16=p_config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()

    print(state_dict)


    with torch.no_grad():

        G.eval()

        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))
            fixed_Gz = F.interpolate(fixed_Gz, size=224)
            VCA_G_z = nn.parallel.data_parallel(VCA, fixed_Gz).view(-1)
        else:
            fixed_Gz = G(fixed_z, G.shared(fixed_y))
            fixed_Gz = F.interpolate(fixed_Gz, size=224)
            VCA_G_z = VCA(fixed_Gz).view(-1)


        print(fixed_Gz.size())
        fixed_Gz_grid = torchvision.utils.make_grid(fixed_Gz.float(), nrow=4, normalize=True)
        print(fixed_Gz_grid.size())
        fixed_Gz_grid = torch.permute(fixed_Gz_grid, (1,2,0))
        neptune_run['train/torch_tensor'].log(File.as_image(fixed_Gz_grid.cpu()))
        neptune_run['train/vca_tensor'].log(VCA_G_z)
        print(fixed_Gz_grid.size())
        fixed_Gz_grid = torchvision.utils.make_grid(fixed_Gz.float(), nrow=4, normalize=False)
        fixed_Gz_grid = torch.permute(fixed_Gz_grid, (1,2,0))
        neptune_run['train/torch_tensor'].log(File.as_image(fixed_Gz_grid.cpu()))
        neptune_run['train/vca_tensor'].log(VCA_G_z)

        image_filename = 'fixed_samples_{}.jpg'.format(1)
        torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                            nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
    
    
    train = train_fns.VCA_generator_training_function(G, VCA, z_, y_, config)

    for epoch in range(state_dict['epoch'], config['num_epochs']):
        print(f"Epoch: {epoch}")

        for i in range(config['iters_per_epoch']):
            state_dict['itr'] += 1

            G.train()
            metrics = train()

            neptune_run["training/batch/loss"].log(metrics['G_loss'])
            neptune_run["training/batch/acc"].log(metrics['VCA_G_z'])
            if not(state_dict['itr'] % config['log_every']):
                print('Epoch: {}    Itr: {}    G_loss: {:.4e}    VCA_G_z: {}'.format(state_dict['epoch'], state_dict['itr'], metrics['G_loss'], metrics['VCA_G_z']))

        
        if config['G_eval_mode']:
            G.eval()
        print("Saving")
        print('Epoch: {}    Itr: {}    G_loss: {:.4e}    VCA_G_z: {}'.format(state_dict['epoch'], state_dict['itr'], metrics['G_loss'], metrics['VCA_G_z']))
        # train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config, experiment_name)
        
        with torch.no_grad():
            G.eval()
            if config['parallel']:
                fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))
            else:
                fixed_Gz = G(fixed_z, G.shared(fixed_y))

            fixed_Gz = F.interpolate(fixed_Gz, size=224)

            VCA_G_z = VCA(fixed_Gz).view(-1)
            fixed_Gz_grid = torchvision.utils.make_grid(fixed_Gz.float(), nrow=4, normalize=True)
            fixed_Gz_grid = torch.permute(fixed_Gz_grid, (1,2,0))
            neptune_run['train/torch_tensor'].log(File.as_image(fixed_Gz_grid.cpu()))
            neptune_run['train/vca_tensor'].log(VCA_G_z)

        
            
        
        state_dict['epoch'] += 1

    # train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config, experiment_name)
    with torch.no_grad():
            G.eval()
            if config['parallel']:
                fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))
            else:
                fixed_Gz = G(fixed_z, G.shared(fixed_y))

            fixed_Gz = F.interpolate(fixed_Gz, size=224)

            VCA_G_z = VCA(fixed_Gz).view(-1)
            fixed_Gz_grid = torchvision.utils.make_grid(fixed_Gz.float(), nrow=4, normalize=True)
            fixed_Gz_grid = torch.permute(fixed_Gz_grid, (1,2,0))
            neptune_run['torch_tensor'].upload(File.as_image(fixed_Gz_grid.cpu()))
            neptune_run['train/vca_tensor'].log(VCA_G_z)

            

def main():

    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    print(config)
    run(config)

if __name__ == '__main__':
    main()