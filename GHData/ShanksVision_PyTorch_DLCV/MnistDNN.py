# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:26:58 2020

@author: shankarj
"""
import torch as pt
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met
import torch.nn.functional as F


class DNN(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = pt.nn.Linear(784, 392)
        self.layer2 = pt.nn.Linear(392, 128)
        self.layer3 = pt.nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        logits = self.layer3(x)        
        return logits
    
    def predict(self, logits):
        preds = F.log_softmax(logits, dim=1).argmax(1, True)        
        return preds
    
def train(model, device, loader, objective, optimizer, epoch, log_interval):
    train_loss=0; train_acc=0
    total_batches = len(loader)
    model.train(True)
    for idx, (data, true_val) in enumerate(loader):
        data, true_val = data.to(device), true_val.to(device)
        input_vec = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        pred_val = model(input_vec)
        loss = objective(pred_val, true_val)
        acc_preds = pt.sum(model.predict(pred_val).view_as(true_val) == true_val)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc_preds.item()
        if idx % log_interval == 0:
            print(f'Epoch : {epoch},  Batch : {idx+1}/{total_batches},'
                 f' Train Loss : {train_loss/(idx+1):.4f}')           
    
    return train_loss/total_batches, float(train_acc)/len(loader.dataset)


def test(model, device, loader, objective):
    val_loss=0; val_acc=0
    model.train(False)
    with pt.no_grad():
        for data, true_val in loader:
            data, true_val = data.to(device), true_val.to(device)
            input_vec = data.view(data.shape[0], -1)            
            pred_val = model(input_vec)
            loss = objective(pred_val, true_val)
            acc_preds = pt.sum(model.predict(pred_val).view_as(true_val) == true_val)
            val_loss += loss.item()
            val_acc += acc_preds.item()
    
    return val_loss/len(loader), float(val_acc)/len(loader.dataset)
    
pt.manual_seed(23)
gpu = pt.device("cuda")

pre_proc = tv.transforms.Compose([tv.transforms.ToTensor(),
                                  tv.transforms.Normalize((0.5), (0.5))])  

train_data = tv.datasets.MNIST('../Data', train=True, transform=pre_proc, 
                               target_transform=None, download=True)
test_data = tv.datasets.MNIST('../Data', train=False, transform=pre_proc, 
                               target_transform=None, download=True)

train_loader = pt.utils.data.DataLoader(train_data, batch_size=300, shuffle=True)
test_loader = pt.utils.data.DataLoader(test_data, batch_size=300)

model = DNN().to(gpu)
optimizer = pt.optim.Adam(model.parameters())
objective = pt.nn.CrossEntropyLoss()
epochs = 20
tloss_history=[]
vloss_history=[]
tacc_history=[]
vacc_history=[]

for i in range(epochs):
    tloss, tacc = train(model, gpu, train_loader, objective, optimizer, i+1, 10)
    vloss, vacc = test(model, gpu, test_loader, objective)
    tloss_history.append(tloss)
    vloss_history.append(vloss)
    tacc_history.append(tacc)
    vacc_history.append(vacc)
    print(f'Epoch : {i+1} => train loss={tloss:.4f}, train acc={tacc:.4f}')
    print(f'............ val loss={vloss:.4f}, val acc={vacc:.4f}')

plt.plot(tloss_history, label='train_loss')
plt.plot(vloss_history, label='val_loss')
plt.title("Loss trend")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(tacc_history, label='train accuracy')
plt.plot(vacc_history, label='val accuracy')
plt.title("Accuracy trend")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Final predictions
pred_labels = np.zeros(len(test_data))
true_labels = np.zeros(len(test_data))
with pt.no_grad():
    for i, (data,labels) in enumerate(test_loader):
        data, labels = data.to(gpu), labels.to(gpu)
        samples = len(data)
        flatvec = data.view(data.shape[0], -1)
        output = model(flatvec)
        preds = model.predict(output).flatten()
        #Copy the data from gpu memory to cpu before passing to numpy
        preds_numpy = preds.cpu().numpy()
        pred_labels[i*samples:(i*samples)+samples] = preds_numpy
        true_labels[i*samples:(i*samples)+samples] = labels.cpu().numpy()

print("Classification Metrics\n----------\n")  
print(met.classification_report(true_labels, pred_labels, 
                                    target_names=test_data.classes)) 