import torch
import torch.nn as nn
from torch.optim import optim, lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import os
import copy

mean=np.array([0.5,0.5,0.5])
std=np.array([0.25,0.25,0.25])

data_transforms={
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    "val":transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
        ])
    
}

data_dir="data/hymneoptera_data"
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                for x in ["train","val"]}
dataloaders={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True, num_workers=0)
             for x in ["train","val"]}
dataset_sizes={x:len(image_datasets[x]) for x in ["train","val"]}
class_names=image_datasets["train"].classes
device=torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
print(class_names)

def imshow(inp,title):
    """imshow for tensor"""
    inp=inp.numpy().transpose((1,2,0))
    inp=std*inp+mean
    inp=np.clip(inp,0,1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

#get a batch of training data
inputs,classes=next(iter(dataloaders["tain"]))

#make a grid from batch
out=torchvision.utils.make_grid(inputs)
imshow(out,title=[class_names[x] for x in classes])

def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since=time.time()
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print("-"*10)
        #each epocn has a training and validation phase
        for phase in ["train","val"]:
            if phase == "train":
                model.train() #set model to training mode
            else:
                model.eval() #set model to evaluate mode
            
            running_loss=0.0
            running_corrects=0
            
            #iterate over data 
            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                
                #forward
                #track history if only in train
                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    _,preds=torch.max(outputs,1)
                    loss=criterion(outputs,labels)
                    
                    #backward +optimize only if in training phase
                    if phase=="train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                #statics
                running_loss +=loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            
            if phase=="train":
                scheduler.step()
            
            epoch_loss=running_loss/dataset_sizes[phase]
            epoch_acc=running_corrects.double()/dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
        time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model