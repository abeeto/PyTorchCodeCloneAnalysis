"""Main function for training, test, etc."""
import json
import logging.config

import PyTorch_VAE.controller.controller
import PyTorch_VAE.utils.utils

# load logging config
with open('log_config.json', 'r') as f:
    log_conf = json.load(f)
logging.config.dictConfig(log_conf)

if __name__ == '__main__':
    PyTorch_VAE.controller.controller.evaluate_model()
    PyTorch_VAE.controller.controller.visualize_training_result()
