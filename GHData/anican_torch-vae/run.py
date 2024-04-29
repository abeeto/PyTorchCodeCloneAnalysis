import argparse
import numpy as np
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
import torch.backends.cudnn as cudnn
import yaml

from models import *
from vae_experiment import VariationalAutoencoderExperiment


def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c', dest="filename", metavar='FILE',
                        help='path to the config file', default='configs/vae.yaml')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    tt_logger = TestTubeLogger(
        save_dir=os.path.join(os.getcwd(), config['logging_params']['save_dir']),
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VariationalAutoencoderExperiment(model, config['exp_params'])

    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     logger=tt_logger,
                     num_sanity_val_steps=5,
                     **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    torch.save(experiment.model.state_dict(), os.path.join(os.getcwd(), 'model.pt'))


if __name__ == '__main__':
    main()
