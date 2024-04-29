'''
VGG-16 with Batch Normalization (From TorchVision)
'''
import torch
import torch.nn as nn
from torchvision import models
from utils import relocate_model

class Model():
    '''
    VGG-16 transfer learning model.
    '''

    def __init__(self, model_download_path, new_model_name, input_feature_size, num_class, pretrained_download=True, freeze_feature_layer=True):
        '''
        Initialize VGG-16 from TorchVision
        '''

        self.vgg_model = models.vgg16_bn(pretrained=pretrained_download, progress=True) #load the model architecture and download the pre-trained weight file if not available already.
        if pretrained_download:
            relocate_model(downloaded_path=model_download_path, model_name=new_model_name) #relocate the downloaded model to the desired path.

        model_params = torch.load(model_download_path+new_model_name) #load the parameters from the pre-trained file.
        self.vgg_model.load_state_dict(model_params) #load the parameters into the architecture.

        for feat_layer in self.vgg_model.features.parameters(): #control the gradient flow in the feature layers.
            feat_layer.requires_grad = not freeze_feature_layer

        print("Model feature parameters have been loaded from the pre-trained weight file.")

        #replace the classifier layers with our desired number of neurons.
        self.vgg_model.classifier = nn.Sequential(
            nn.Linear(input_feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_class)
        )

    def __call__(self):
        '''
        Returns the instantiated model.
        '''

        return self.vgg_model
