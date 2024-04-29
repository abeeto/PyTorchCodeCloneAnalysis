import logging
import os
from logging.config import fileConfig

import click
import toml

from your_project.eval.evaluate_function import evaluate_model


@click.command()
@click.option('--config', '-c', help='config file for evaluation parameters')
def run(config):
    config = toml.load(config)
    logging.info("> evaluating model")
    evaluate_model(config)


if __name__ == '__main__':
    logs_dir = os.path.join('.', 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # initialize logging from ini file
    fileConfig('logging_config.ini')
    # run from config in toml file
    run()
