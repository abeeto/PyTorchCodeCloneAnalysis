import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from vca_utils import load_checkpoint

import utils

def run(config):
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device ${device}")

    torch.backends.cudnn.benchmark = True

    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))
    print("Experiment name is {}".format(experiment_name))

    G = model.Generator(**config).to(device)
    D = None
    G_ema, ema = None, None

    VCA = Amy_IntermediateRoad( lowfea_VGGlayer=10, highfea_VGGlayer=36, is_highroad_only=False, is_gist=False)
    print("Loading VCA from {}".format(config['vca_filepath']))
    VCA = load_checkpoint(VCA, config['vca_filepath'])
    VCA = VCA.to(device)

    state_dict = {
        'itr': 0,
        'epoch': 0,
        'save_num': 0,
        'save_best_num': 0,
        'best_IS': 0,
        'best_FID': 999999,
        'config': config
    }

    print("Loading Generator weights")
    utils.load_weights(G, D, state_dict,
                            config['load_weights_root'], experiment_name,
                            config['load_weights'] if config['load_weights'] else None,
                            G_ema if config['ema'] else None)

    state_dict['config'] = config

    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()

    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))
        else:
            fixed_Gz = G(fixed_z, G.shared(fixed_y))

        fixed_Gz = F.interpolate(fixed_Gz, size=224)

        VCA_G_z = VCA(fixed_Gz).view(-1)

    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.makedirs('%s/%s' % (config['samples_root'], experiment_name))

    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)

    print(VCA_G_z)

def main():

    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    print(config)
    run(config)

if __name__ == '__main__':
    main()