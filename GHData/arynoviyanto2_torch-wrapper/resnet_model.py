import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResnetModel(nn.Module):

    def __init__(self, num_targets, dataset_name = ''):
        super(ResnetModel, self).__init__()

        self.dataset_name = dataset_name
        self.num_targets = num_targets

        self.model = torchvision.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features, num_targets)

    def forward(self, batch_features): 
        return F.log_softmax(self.model(batch_features), dim=1)

    def get_name(self):
        return 'resnet_' + self.dataset_name