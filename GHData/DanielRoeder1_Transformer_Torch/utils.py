import numpy as np
import time
import json

def get_praram_count(model) -> int:
    """
    Returns the number of trainable parameters in the model
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params

def get_time():
    return time.strftime("%H:%M:%S",  time.localtime())

class AverageMeter():
    def __init__(self) -> None:
        self.count = 0 
        self.loss = 0
        self.accuracy = 0
    def update(self, loss, accuracy, count = 1):
        self.loss += loss * count
        self.accuracy  += accuracy * count
        self.count += count
    def get_avg(self):
        return self.loss / self.count, self.accuracy / self.count

class ConfigObject(object):
    """
    Creats Class object based on json / dict
    """
    def __init__(self, config_path) -> None:
        config_dict = json.load(open(config_path, "r"))
        self.update(config_dict)
    def update(self, input_dict):
        for key in input_dict:
            setattr(self, key, input_dict[key])