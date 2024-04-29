import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from model.lit_resnet import LitResnet
from data.get_data import cifar10_dm
from results.results import plt_metrics

import configs.load_configs as config

if __name__ == '__main__':
    cfg = config.load_config()

    model = LitResnet(cfg[0]['model']['lit_resnet']['lr'])

    trainer = Trainer(
        max_epochs=cfg[0]['model']['lit_resnet']['max_epochs'],
        accelerator=cfg[0]['model']['lit_resnet']['acc'],
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir=cfg[2]['logs']['log_dir']),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    plt_metrics(trainer)


