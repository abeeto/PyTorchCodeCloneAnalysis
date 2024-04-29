import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='./configs/cvae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
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
    experiment = VAEXperiment(model,
                              config['exp_params'])

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename=config['exp_params']['dataset'] + '-{epoch:02d}-{val_loss:.2f}', save_top_k=-1)

    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     checkpoint_callback=True,
                     min_epochs=1,
                     logger=tt_logger,
                     log_every_n_steps=100,
                     limit_train_batches=1.,
                     val_check_interval=1.,
                     num_sanity_val_steps=5,
                     callbacks=[checkpoint_callback],
                     **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    print(config['exp_params']['LR'])
    runner.fit(experiment)
