import json
import logging.config


def load_settings():
    with open("logging.cfg", "r") as stream:
        logging_config = json.load(stream)
    logging.config.dictConfig(logging_config)