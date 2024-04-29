#import the necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F 
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import numpy as np

import time
import os
import copy
import matplotlib.pyplot as plt

import random
import scipy.misc
from PIL import Image
from torch.autograd import Variable


#Load Data
# Data augmentation and normalization for training
# Just normalization for validation
#Define the model
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}
#Assumption is you have train, val, and test folders, where within each folder
#You have Mask and NoMask folders as well.

data_dir = 'Mask_NoMask/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=4)
            for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
class_names = image_datasets['train'].classes

device = "cuda:0" if torch.cuda.is_available() else "cpu"

inv_normalize = transforms.Normalize(
 mean=[-0.4302/0.2361, -0.4575/0.2347, -0.4539/0.2432],
 std=[1/0.2361, 1/0.2347, 1/0.2432]
)

#classes = train_data.classes
classes = image_datasets['test'].classes
#encoder and decoder to convert classes into integer
decoder = {}
for i in range(len(classes)):
 decoder[classes[i]] = i
encoder = {}
for i in range(len(classes)):
 encoder[i] = classes[i]

def test(model,dataloader,device,batch_size):
    running_corrects = 0
    running_loss=0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    sm = nn.Softmax(dim = 1)
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(dataloaders['test']):
        data, target = Variable(data), Variable(target)
        data = data.type(torch.FloatTensor).to(device)
        target = target.type(torch.LongTensor).to(device)
        model.eval()
        output = model(data)
        loss = criterion(output, target)
        output = sm(output)
        _, preds = torch.max(output, 1)
        running_corrects = running_corrects + torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        data = data.cpu().numpy()

        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            if(preds[i]!=target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])

    epoch_acc = running_corrects.double()/(len(dataloaders['test'])*batch_size)
    epoch_loss = running_loss/(len(dataloaders['test'])*batch_size)
    print('Testset Accuracy {}'.format(epoch_acc))
    print('Testset Loss {}'.format(epoch_loss))
    return true,pred,image,true_wrong,pred_wrong

#plot wrong predictions of the model 
def wrong_plot(n_figures,true,ima,pred,encoder,inv_normalize):
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures/3)
    fig,axes = plt.subplots(figsize=(14, 10), nrows = n_row, ncols=3)
    i=0
    for ax in axes.flatten():
        a = random.randint(0,len(true)-1)

        image,correct,wrong = ima[a],true[a],pred[a]
        image = torch.from_numpy(image)
        correct = int(correct)
        c = encoder[correct]
        wrong = int(wrong)
        w = encoder[wrong]
        f = 'A:'+c + ',' +'P:'+w
        if inv_normalize !=None:
            image = inv_normalize(image)

        image = image.numpy().transpose(1,2,0)
        i += 1
        im = ax.imshow(image)
        ax.set_title(f)
        ax.axis('off')
    plt.show()

if __name__ == "__main__":
    #Load the pretrained model
    model_ft = torch.load('model.h5')
    #Test the model
    true,pred,image,true_wrong,pred_wrong = test(model_ft, dataloaders['test'], device,batch_size=4)
    #Visualize wrong prediction, if any.
    wrong_plot(12, true_wrong, image, pred_wrong, encoder, inv_normalize)