import argparse

parser = argparse.ArgumentParser(description="image path")

parser.add_argument('-p', '--path', type=str)
args = parser.parse_args()

path = args.path

import os
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

batch_size = 20
num_workers = 0

data_directory = 'C:\\Users\\vp393001\\Desktop\\dogImages\\'
train_directory = os.path.join(data_directory, 'train\\')
valid_directory = os.path.join(data_directory, 'valid\\')
test_directory = os.path.join(data_directory, 'test\\')

standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     standard_normalization]),
                   'val': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     standard_normalization]),
                   'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                     transforms.ToTensor(), 
                                     standard_normalization])
                  }
train_data = datasets.ImageFolder(train_directory, transform=data_transforms['train'])
valid_data = datasets.ImageFolder(valid_directory, transform=data_transforms['val'])
test_data = datasets.ImageFolder(test_directory, transform=data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)
loaders_scratch = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}

## TODO: Specify data loaders
loaders_transfer = loaders_scratch.copy()

import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 
model_transfer = models.resnet50(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad = False
model_transfer.fc = nn.Linear(2048, 133, bias=True) 
fc_parameters = model_transfer.fc.parameters()
for param in fc_parameters:
    param.requires_grad = True
model_transfer = model_transfer.cuda()

import torch.optim as optimization
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optimization.SGD(model_transfer.fc.parameters(), lr=0.001)

### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
from PIL import Image
import torchvision.transforms as transforms

class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]
model_transfer.load_state_dict(torch.load('C:\\Users\\vp393001\\Downloads\\model_transfer.pt'))
loaders_transfer['train'].dataset.classes[:10]
class_names[:10]
def load_input_image(img_path):    
    image = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     standard_normalization])
    
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)    
    return image
def predict_breed_transfer(model, class_names, img_path):
    # load the image and return the predicted breed    
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()    
    idx = torch.argmax(model(img))    
    return class_names[idx]

img_path = path    
predition = predict_breed_transfer(model_transfer, class_names, img_path)
print("image_file_name: {0}, \t predition breed: {1}".format(img_path, predition))


    
  