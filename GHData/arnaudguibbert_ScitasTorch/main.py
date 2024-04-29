import logging.config
import yaml
import os
from datetime import datetime

from argparse import ArgumentParser

from models import CNN, TrainModel, SetOptimizer
from scripts import AutoScript
from data import SetDataLoader

import torch.nn as nn

with open("config.yml","r") as file:
    config = yaml.safe_load(file)
    logging.config.dictConfig(config["logger"])

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = ArgumentParser("main")

    logger.info("Model settings : %s",config["model"])
    logger.info("Training settings : %s",config["train"])
    logger.info("Optimizer settings : %s",config["optimizer"])
    logger.info("DataLoader settings : %s",config["data"])
    logger.info("Checkpoint available : %s",config["checkpoint"] is not None)

    model = CNN(**config["model"])
    optimizer = SetOptimizer(model,**config["optimizer"])
    loader = SetDataLoader(**config["data"])
    trainer = TrainModel(model,
                         optimizer,
                         criterion=nn.CrossEntropyLoss(),
                         train_data=loader,**config["train"])

    limit_reached = trainer.train()

    if limit_reached:
        logger.info("Relaunching a script")
        checkpoint_path = os.path.join("checkpoints","model" + str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
        trainer.export(checkpoint_path)
        autoscripter = AutoScript(config["launch_file"],"config.yml")
        autoscripter.modify_config(checkpoint_path)
        autoscripter.launch_scitas_script()
