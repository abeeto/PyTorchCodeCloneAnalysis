#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin

from vilt.config import ex
from vilt.modules.mmsp_module import ViLTransformerSS
from vilt.datamodules.light_datamodule import LightDataModule
import sys


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])  # 设置随机种子，方便复现

    dm = LightDataModule(_config, dist=True)


    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    log_dir = f'{_config["workspace_dir"]}/{_config["log_dir"]}'

    os.makedirs(log_dir, exist_ok=True)


    logger = pl.loggers.TensorBoardLogger(
        log_dir,
        name=f'{exp_name}_seed{_config["seed"]}{"_test" if _config["test_only"] else ""}',
        # default_hp_metric=False,
        # log_graph=True,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    earlystopping_callback = pl.callbacks.EarlyStopping(
        monitor='hp_metric',
        patience=_config["patience"],
        mode=_config["mode"])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="hp_metric",
        mode=_config["mode"],
        save_last=True,
    )
    
    callbacks = [checkpoint_callback, lr_callback, earlystopping_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    # 计算每一batch size需要多少grad_steps
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # refer to https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.html
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],  # num of gpu nodes
        precision=_config["precision"],  # float precision, can be 64, 32 or 16
        accelerator="ddp",
        benchmark=True, # if true, enables cudnn.benchmark, 加速计算
        deterministic=True,  # if true, enables cudnn.deterministic, 返回确定卷积算法，方便复现结果
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,  # max gradient steps
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,  # 累加多个batches的grad
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],  # 快速运行n or 1 batch 来找bug
        val_check_interval=_config["val_check_interval"],  # how often to check the validation set
        # plugins=[DDPPlugin(find_unused_parameters=True)],  # 没调通
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm, verbose=True)
