import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Models are subclasses of nn.Module and:
#   - Contain submodules (layers) as member variables
#   - Implement a 'forward' function that propagates the input through the network
class Model(nn.Module):

    # __init__ can take any number of parameters
    def __init__(self, in_channels, n_classes):
        super().__init__()

        # Member modules can be grouped together into larger units using either:
        #   - nn.Sequential
        #   - nn.ModuleList
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Member modules don't have to be grouped together at all. They can just be member variables
        self.maxpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=128, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=n_classes)

        # Note: for a module's weights to be automatically included in the Model's
        # trainable parameters, it must be declared either as a direct member variable, or
        # it must be within a nn.Sequential/nn.ModuleList member variable.
        # For instance, if you write:
        #   self.layers = [nn.Conv2d(...), nn.Linear(...)]
        # PyTorch will not register these layers' weights as parameters of the Model,
        # even if you use the layers in the forward pass.

    # forward can take any number of parameters, and can return any number of values
    #
    # You can have ANY Python code and control flow you want within the forward function (if's, for's, etc.)
    # PyTorch creates graphs dynamically for each forward pass, so your model doesn't have to do the same thing
    # every time. This is particularly useful for things like RNN's
    def forward(self, image):
        x = self.features(image)
        x = self.maxpool(x)

        x = x.view(-1, 128)     # Flatten height and width dimensions

        x = self.fc1(x)
        x = self.fc2(x)

        # When training, the PyTorch cross-entropy loss adds a logsoftmax layer to your output,
        # so you should just return the raw logits
        if not self.training:
            x = F.softmax(x, dim=1) # The nn.functional library provides many parameterless layers

        return x


class PretrainedModel(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        alexnet = models.alexnet(pretrained=True)
        alexnet.features[-1] = nn.AdaptiveMaxPool2d(output_size=1)  # Make alexnet work with any sized input
        self.features = alexnet.features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=n_classes)
        )

    def forward(self, image):
        x = self.features(image)
        x = x.view(-1, 256)
        x = self.classifier(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x

    # If you don't want to fine-tune the whole model, you simply
    # don't have to return all of the model's parameters to the optimizer.
    # In this case, I'm overriding the base class's existing `parameters()` function for simplicity
    def parameters(self):
        return self.features.parameters()
