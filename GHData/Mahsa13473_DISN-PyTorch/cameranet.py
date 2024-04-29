import torch
import torch.nn as nn
import pdb

import numpy as np
import matplotlib.pyplot as plt

import torchvision.models as models
#from torchsummary import summary
import torch.nn.functional as F


class cameranet(nn.Module):
    def __init__(self): #,configs
        super(cameranet, self).__init__()

        encoder = models.vgg16(pretrained = True)
        self.fc = nn.Linear(7*7*512, 1024)

        self.global_features = nn.Sequential(*list(encoder.children())[:-2])

        # Rotation branch
        self.layer1_rotation = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.layer2_rotation = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.layer3_rotation = nn.Sequential(nn.Linear(256, 6))

        self.rotation_branch = nn.Sequential(self.layer1_rotation, self.layer2_rotation, self.layer3_rotation)

        # Translation branch
        self.layer1_translation = nn.Sequential(nn.Linear(1024, 128), nn.ReLU())
        self.layer2_translation = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.layer3_translation = nn.Sequential(nn.Linear(64, 3))

        self.translation_branch = nn.Sequential(self.layer1_translation, self.layer2_translation, self.layer3_translation)


    def forward(self, img):

        global_feat = self.global_features(img)
        global_feat = global_feat.view(-1, 7*7*512)
        global_feat = F.relu(self.fc(global_feat))

        rotation_pred = self.rotation_branch(global_feat)
        translation_pred = self.translation_branch(global_feat)

        return rotation_pred, translation_pred






if __name__ == '__main__':
    model = cameranet()
    print(model)
    #summary(model, input_size = ((3, 1), (3,224,224)), device = 'cpu')
    #summary(model, input_size = (3,224,224), device = 'cpu')

    batch_size = 1
    image = torch.randn(batch_size, 3, 224, 224)

    R, T = model(image)

    print(R.shape)
    print(T.shape)
