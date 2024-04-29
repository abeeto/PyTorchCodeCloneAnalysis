from collections import OrderedDict

import torch

from model import UAEModel


def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        else:
            new_dict[k] = v
    target.load_state_dict(new_dict)


def load_model(max_len, device):
    model = UAEModel(max_len, device).to(device)
    load_weights(model, torch.load('model.bin', map_location='cpu'))
    return model
