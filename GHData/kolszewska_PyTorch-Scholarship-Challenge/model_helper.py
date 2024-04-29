import torch
from torch import nn
from torchvision import models

import classifier


def get_model_for_training() -> nn.Module:
    """Get model for training"""
    vgg_model = models.vgg19(pretrained=True)
    custom_classifier = classifier.Classifier()
    model = apply_classifier_to_model(vgg_model, custom_classifier)
    model.cuda()
    return model


def save_model(model, model_path='model.pt') -> None:
    """Save model state to file.

    :param model model that the state should be saved to a file
    :param model_path path that model state should be saved to
    """
    torch.save(model.state_dict(), model_path)


def load_model_from_file(model_path='model.pt') -> nn.Module:
    """Load model from file.

    :param model_path path that model state should be loaded from
    """
    custom_model = models.vgg19(pretrained=True)
    custom_classifier = classifier.Classifier()
    custom_model.classifier = custom_classifier
    custom_model.load_state_dict(model_path, strict=False)
    custom_model.eval()
    return custom_model


def apply_classifier_to_model(custom_classifier: nn.Module, pretrained_model: nn.Module) -> nn.Module:
    """Apply custom classifier to existing pretrained model.

    :param custom_classifier custom classifier
    :param pretrained_model already existing model, default VGG19
    """
    model = pretrained_model

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = custom_classifier
    return model
