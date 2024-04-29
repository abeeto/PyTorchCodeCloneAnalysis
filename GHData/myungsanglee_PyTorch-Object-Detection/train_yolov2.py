import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from torchinfo import summary

from dataset.detection.yolo_dataset import YoloDataModule
from module.yolov2_detector import YoloV2Detector
from models.detector.yolov2 import YoloV2
from utils.utility import make_model_name
from utils.module_select import get_model
from utils.yaml_helper import get_configs


def train(cfg):
    data_module = YoloDataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size']
    )

    backbone_features_module = get_model(cfg['backbone'])(
        pretrained=cfg['backbone_pretrained'], 
        devices=cfg['devices'],
        features_only=True,
        out_indices=[4, 5]
    )
    
    model = YoloV2(
        backbone_features_module=backbone_features_module,
        num_classes=cfg['num_classes'],
        num_anchors=len(cfg['scaled_anchors'])
    )
    
    summary(model, input_size=(1, cfg['in_channels'], cfg['input_size'], cfg['input_size']), device='cpu')

    model_module = YoloV2Detector(
        model=model, 
        cfg=cfg
    )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor='val_loss', 
            save_last=True,
            every_n_epochs=cfg['save_freq']
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            verbose=True
        )
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], make_model_name(cfg), default_hp_metric=False),
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        **cfg['trainer_options']
    )
    
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    train(cfg)
    