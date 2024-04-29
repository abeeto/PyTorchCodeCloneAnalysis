# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:06:54 2018

@author: vprayagala2

MNIST Fashon Data set
"""
#%%
import torch
from torch import utils
from torch import nn , optim
import torch.nn.functional as F
from torchvision import datasets, transforms
#%%
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),
                                                   (0.5,0.5,0.5))
                              ])
#Download the mnist data
train_data=datasets.FashionMNIST("C:\Data\MNIST-Data\\",download=True,train=True,
                          transform=transform)
train_loader=utils.data.DataLoader(train_data,
                                         batch_size=64,
                                         shuffle=True)


print("Train Image Shape:{}".format(train_loader.dataset.train_data.shape))
print("Train Label Shape:{}".format(train_loader.dataset.train_labels.shape))

test_data=datasets.FashionMNIST("C:\Data\MNIST-Data\\",
                                download=False,
                                train=False,
                                transform=transform)

test_loader=utils.data.DataLoader(test_data,shuffle=True)

print("Test Image Shape:{}".format(test_loader.dataset.test_data.shape))
print("Test Label Shape:{}".format(test_loader.dataset.test_labels.shape))
#%%
#Define Network Topology
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)
        
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self,x):
        x=x.view(x.shape[0],-1)
        
        h1 = self.dropout(F.relu(self.fc1(x)))
        h2 = self.dropout(F.relu(self.fc2(h1)))
        h3 = self.dropout(F.relu(self.fc3(h2)))
        y = F.log_softmax(self.fc4(h3),dim=1)
        
        return y
#%%
model=Classifier()
criterion = nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),lr=0.003)
#%%
#Train the network
epochs=15
stesp = 0

train_loss_l , test_loss_l = [] , []
for i in range(epochs):
    running_loss = 0
    for image, label in train_loader:
        #train_img=image.view(image.shape[0],-1)
        
        #Forward Pass
        logps = model.forward(image)
        train_loss = criterion(logps,label)
        optimizer.zero_grad()
        train_loss.backward()
        
        optimizer.step()
        running_loss += train_loss.item()
        #print("Current Epoch loss:{}".format(loss))
        #print("Running Loss:{}".format(running_loss))
    else:
        test_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            model.eval()
            
            for image, label in test_loader:
                log_ps=model(image)
                test_loss += criterion(log_ps,label)
                
                ps=torch.exp(log_ps)
                pred_p,pred_class = ps.topk(1,dim=1)
               
                equals = pred_class == label.view(*pred_class.shape)
                accuracy += torch.mean (equals.type(torch.FloatTensor))
        
        model.train()
        
        train_loss_l.append(running_loss/len(train_loader))
        test_loss_l.append(test_loss/len(test_loader))
        
        print("Epchs {}/{}..".format(i+1 , epochs),
              "Training Loss:{}..".format(running_loss/len(train_loader)),
              "Testing Loss:{}..".format(test_loss/len(test_loader)),
              "Accuracy:{}".format(accuracy/len(test_loader)))
#%%
import matplotlib.pyplot as plt

plt.plot(train_loss_l,label="Training Losses")
plt.plot(test_loss_l,label="Testing Losses")
plt.legend(frameon=False)
#%%
#Do predictions
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from pprint import pprint

pred_label, true_label = [], []
with torch.no_grad():
    model.eval()
    
    for image, label in test_loader:
        log_ps=model(image)
        test_loss += criterion(log_ps,label)
        
        ps=torch.exp(log_ps)
        pred_p,pred_class = ps.topk(1,dim=1)
        pred_label.append(pred_class)
        true_label.append(label)
                

res = classification_report(pred_label,true_label)
conf = confusion_matrix(pred_label,true_label)
acc = accuracy_score(pred_label,true_label)

pprint("Classification Report:\n{}".format(res))
pprint("Confusion Matrix:\n{}".format(conf))
pprint("Accuracy Score:{}".format(acc))
#%%