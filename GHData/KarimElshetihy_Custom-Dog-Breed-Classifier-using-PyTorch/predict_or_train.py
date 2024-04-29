# import libraries
# Imports here
import time
import torch
from torch import nn, optim
import helper
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
from get_input_args import get_input_args
from train import load_data, validate, train, plot_stats, save_model
from predict import load_model, predict, process_image, imshow, view_class
from get_input_args import get_input_args

# --------------------------------------------------------
# args input
args = get_input_args()
dir = args.dir
arch = args.arch
process = args.process
learn_rate = float(args.learn_rate)
layers = int(args.layers)
epochs = int(args.epochs)
gpu = args.gpu
# --------------------------------------------------------
# Models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()

# Models dict
models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# apply model to input
model = models[arch]
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learn_rate)
# --------------------------------------------------------

def main(model, epochs, optimizer, criterion):
    
    train_loader, validation_loader, test_loader, Labels_df = load_data(dir)
    
    if process == 'train':
        USE_GPU = True if gpu == 'on' else False
        if USE_GPU and torch.cuda.is_available():
            print('using device: cuda')
        else:
            print('using device: cpu')

        # put model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        Training_Loss, Validation_Loss, Validation_Accuracy, Validation_Corrects = train(model, train_loader, validation_loader, epochs, optimizer, criterion)
        
        plot_stats(epochs, Training_Loss, Validation_Loss, Validation_Accuracy)

        path = '~/opt'
        save_model(model, path, epochs=epochs, arch=arch)
    
    else:
        path = '~/opt/Classifier.pth'
        classifier = models[model]
        classifier, optimizer, epochs, criterion = load_model(classifier, path)
        # # put model on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        classifier = classifier.to(device)

        probes = predict(model, dir, Labels_df)

# Call to main function to run the program
if __name__ == "__main__":
    main()

