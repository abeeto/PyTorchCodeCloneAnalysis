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

def view_class(probes, names):
    ''' Function for viewing an image and it's predicted classes.
    '''
    probes = probes.cpu().data.numpy().squeeze()
    names['probes'] = probes
    highest = names.nlargest(5, ['probes'])[['probes','flower_name']]
    names = names.sort_values(by=['probes'])
    
    x = highest['probes']
    y = highest['flower_name']

    plt.figure(figsize=(15,5));
    plt.barh(y=y, width=x*100);
    plt.yticks(size=14);

    return names, highest

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.permute(2, 3, 1, 0).squeeze(3);
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image*std + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1);
    
    plt.imshow(image);


def process_image(model, image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # load the image
    image_PIL = Image.open(image_path) if isinstance(image_path, str) else image_path

    # define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    image_tensor = transform(image_PIL)
    
    # resize the tensor (add dimension for batch)
    image_tensor.unsqueeze_(0)
    
    # address tensor as output not wrapper
    image_tensor.requires_grad_(False)

    # apply model to input
    model = model

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()

    output = model(image_tensor)

    # return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()
    
    return image_tensor


def predict(model, image_path, names):
    # model.eval() is a kind of switch for some specific
    # layers/parts of the model that behave differently 
    # during training and inference (evaluating) time.
    
    model.eval()
    
    probs = 0
    
    image = process_image(model, image_path)
    imshow(image, ax=None, title=None)
    
    label = image_path.split('/')[-2]

    # Calculate the class probabilities (softmax) for image
    
    with torch.no_grad():
        output = model.forward(image)

        # Calculating Accuracy
        _, preds = torch.max(output.data, 1)
        Correct = "Correct" if (preds == label) else "Incorrect"

    probs = torch.exp(output)
    # Plot the image and probabilities
    names, highest = view_class(probs, names)
    Actual_Name = names.loc[names['Id'] == int(label)+1]['flower_name'].item()
    Pred_Name = highest.nlargest(1, 'probes')['flower_name'].item()
    Accuracy = np.round(100*highest.nlargest(1, 'probes')['probes'].item(), decimals=2)


    print("# Prediction is: {} \n# Accuracy: {}%".format(Correct, Accuracy))
    print("# Prediction Label: {} \n# Actual Label: {}".format(Pred_Name, Actual_Name))

    return probs


def load_model(model, path):
    # Checking if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize the model class before loading the model
    classifier = model

    # Load the model checkpoint
    checkpoint = torch.load(path, map_location=device)

    # Load model weights state
    classifier.load_state_dict(checkpoint['Classifier_state'])

    # Load trained optimizer state
    optimizer.load_state_dict(checkpoint['Optimizer_state'])

    # Load number of previous epochs
    epochs = checkpoint['Epoch']

    # load the criterion
    criterion = checkpoint['Loss_Function']

    return classifier, optimizer, epochs, criterion




