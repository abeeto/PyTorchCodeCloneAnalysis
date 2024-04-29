import os

from torchvision import transforms
import pytorch_lightning as pl 
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from data.data import MyDataModule as ClassifyDatasets
from models.model import MutiCls_Classify as ClassifyModel

from utils.utils import get_yaml


def train(yaml_file):

    YAML_FILE = yaml_file
    params = get_yaml(YAML_FILE)

    os.makedirs(params['save_path'], mode=0o777, exist_ok=True)
    logger = TensorBoardLogger(params['save_path'], name='')

    # log dir
    params['log_dir'] = logger.log_dir

    # weights dir
    weight_path = os.path.join(logger.log_dir, 'weights')
    params['weights_dir'] = weight_path

    # datasets
    dm = ClassifyDatasets(params)
    
    # model 
    # print(params)
    model = ClassifyModel(hparams = params)

    trainer = pl.Trainer(
        max_epochs=params['max_epoch'], 
        accelerator=params['accelerator'], 
        devices=params['num_gpus'],
        logger=logger,
        precision=params['precision'],
        limit_train_batches=1.0,
        enable_checkpointing=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        val_check_interval=1.0,
        max_steps=-1,
        strategy="ddp_find_unused_parameters_false"
    )

    trainer.fit(model, datamodule=dm)
    return trainer, model, dm


if __name__ == "__main__":
    
    seed_everything(42)
    
    yaml_file = './config/classify.yaml'
    train(yaml_file)
