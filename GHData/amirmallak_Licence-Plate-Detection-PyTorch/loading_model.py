import torch
import config

from neural_network_model import LicensePlateDetectionNN


def load_model() -> LicensePlateDetectionNN:
    # Loading our model
    model = LicensePlateDetectionNN()
    model_path = config.model_path
    model.load_state_dict(torch.load(model_path))

    # Calling eval() after loading our model for setting dropout and batch normalization layers to evaluation mode
    # before running inference. Failing to do this will yield inconsistent inference results.
    model.eval()

    return model
