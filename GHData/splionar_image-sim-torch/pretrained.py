import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import random
from layers import Unet
import torch.nn as nn
import torch.nn.functional as F

# convert data to torch.FloatTensor
transform = transforms.ToTensor()
train_data = datasets.ImageFolder(root = '/content/drive/My Drive/food_224_55ktrain', transform = transform)

# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 32

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

'''
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
'''
model = models.resnet18(pretrained=True)

# Freeze layers ####################
for param in model.parameters():
    param.requires_grad = False
####################################
    
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1000)
#model.fc = Identity()
model.cuda()



# specify loss function
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
                             #, weight_decay=0.001)
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)


# number of epochs to train the model
n_epochs = 10000
ep = 1
for epoch in range(1, n_epochs+1):
    
    # monitor training loss
        
    ###################
    # train the model #
    ###################
    it = 0
    train_loss = 0.0
    for data in train_loader:

        images, _ = data
        images = images.to('cuda')
        optimizer.zero_grad()
        l= model(images[:,:,:,:224])
        m = model(images[:,:,:,224:448])
        r = model(images[:,:,:,448:])

        # loss
        triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)
        loss = triplet_loss(l,m,r)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
        
        it = it + 1
        
        if it%200 == 0:
            print("Iteration: {} Loss: {}".format(it,loss))

        if it%1000 == 0:
            #print('Saving model')
            torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/pretrained/resnet18.pt")
          
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    
    print('Saving model')
    
    if ep%5 == 0:
        torch.save(model.state_dict(), "/content/drive/My Drive/IML/task4/pretrained/resnet18_epoch{}.pt".format(ep))
    
    ep = ep + 1
