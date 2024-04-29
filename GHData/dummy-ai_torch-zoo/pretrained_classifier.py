import os
import json
import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision.transforms import Normalize
from PIL import Image
import numpy as np
import requests

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

# map human-friendly names to actual module name
MODEL_NAMES = {
    'AlexNet': 'alexnet',
    'ResNet-18': 'resnet18',
    'ResNet-34': 'resnet34',
    'ResNet-50': 'resnet50',
    'ResNet-101': 'resnet101',
    'ResNet-152': 'resnet152',
    'VGG-11': 'vgg11_bn',
    'VGG-13': 'vgg13_bn',
    'VGG-16': 'vgg16_bn',
    'VGG-19': 'vgg19_bn',
    'Inception-v3': 'inception_v3',
    'SqueezeNet': 'squeezenet1_1',
    'DenseNet-121': 'densenet121',
    'DenseNet-161': 'densenet161',
    'DenseNet-169': 'densenet169',
    'DenseNet-201': 'densenet201',
}


class PretrainedClassifier():
    def __init__(self, model_name):
        if model_name in MODEL_NAMES:
            model_name = MODEL_NAMES[model_name]
        else:
            assert model_name in MODEL_NAMES.values(), MODEL_NAMES
        self.net = getattr(models, model_name)(pretrained=True)
        self.labels = self.process_labels()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])


    def process_labels(self, download=False):
        if download:
            labels = requests.get(LABELS_URL).json()
        else:
            with open('labels.json') as fp:
                labels = json.load(fp)
        return {int(key): value for key, value in labels.items()}


    def softmax_to_label(self, softmax, to_name=True):
        assert isinstance(softmax, Variable)
        _, argmax = softmax.max(1)
        label = int(np.asscalar(argmax.data.numpy()))
        if to_name:
            return self.labels[label]
        else:
            return label


    def predict(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        assert isinstance(img, Image.Image)
        img = img.resize((224, 224))
        img = np.array(img)
        img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        img = img[None] # expand a fake batch dim
        img = img.astype(np.float32) / 255.
        img = self.normalize(torch.FloatTensor(img))
        prediction = self.net(Variable(img, volatile=True))
        return self.softmax_to_label(prediction)


if __name__ == '__main__':
    classifier = PretrainedClassifier('SqueezeNet')
    for img in os.listdir('demo_imgs'):
        img = os.path.join('demo_imgs', img)
        print('file {} -> {}'.format( img, classifier.predict(img) ))
