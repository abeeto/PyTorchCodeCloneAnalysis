import logging
import os
from logging.config import fileConfig

import click
import toml

from your_project.train_model import train_model


@click.command()
@click.option('--config', '-c', help='config file for experimentation parameters')
def run(config):
    config = toml.load(config)
    logging.info("> training model")
    train_model(config)


if __name__ == '__main__':
    logs_dir = os.path.join('.', 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # initialize logging from ini file
    fileConfig('logging_config.ini')
    # run from config in toml file
    run()
