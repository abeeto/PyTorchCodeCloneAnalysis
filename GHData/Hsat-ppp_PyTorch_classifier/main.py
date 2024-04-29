"""Main function for training, test, etc."""
import json
import logging.config

import PyTorch_classifier.controller.classifier_controller
import PyTorch_classifier.utils.utils

# load logging config
with open('log_config.json', 'r') as f:
    log_conf = json.load(f)
logging.config.dictConfig(log_conf)

if __name__ == '__main__':
    PyTorch_classifier.controller.classifier_controller.evaluate_model()
