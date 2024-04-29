import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F

import helper
from models.model import Classifier



#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

# general transform
data_transform = transforms.Compose([transforms.Resize((28, 28)),  # resize all images into 28x28 pixels
                                     transforms.Grayscale(num_output_channels=1), # turns 3 channel into 1 (grey scale)
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])

# upload your test data and apply general transforms
test_dataset = datasets.ImageFolder(root='data/test_dataset/',
                                           transform=data_transform)

# data loader
testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=64, shuffle=True,
                                             num_workers=1)


# Train the network ------------------------------------------------------------
# Create the network, define the criterion and optimizer
model = Classifier()

# Test out your small_test_dataset!
state_dict = torch.load('models/checkpoint_1.pth')
model.load_state_dict(state_dict)

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]
#helper.imshow(img)

# Calculate the class probabilities (softmax) for img
ps = torch.exp(model(img))

# Plot the image and probabilities
helper.view_classify(img, ps, version='F')
