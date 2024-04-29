# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:57:52 2020

@author: shankarj
"""
#there is a bug in mapping of class labels to image folders. need to fix this

import torch as pt
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met
import torch.nn.functional as F
import sys
import pytorch_model_summary as summary
    
def predict(logits):
        preds = F.log_softmax(logits, dim=1).argmax(1, True)        
        return preds
    
def train(model, device, loader, objective, optimizer, epoch, log_interval):
    train_loss=0; train_acc=0
    total_batches = len(loader)
    model.train(True)
    for idx, (data, true_val) in enumerate(loader):
        data, true_val = data.to(device), true_val.to(device)        
        optimizer.zero_grad()
        pred_val = model(data)
        loss = objective(pred_val, true_val)
        acc_preds = pt.sum(predict(pred_val).view_as(true_val) == true_val)
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
            pred_val = model(data)
            loss = objective(pred_val, true_val)
            acc_preds = pt.sum(predict(pred_val).view_as(true_val) == true_val)
            val_loss += loss.item()
            val_acc += acc_preds.item()
    
    return val_loss/len(loader), float(val_acc)/len(loader.dataset)

def plot_loss_wrong_preds(history, x=None, y=None, yhat=None, labels=None):
    #Plot the train loss and val loss
    plt.plot(history[0], label='train_loss')
    plt.plot(history[1], label='val_loss')
    plt.title('Loss trend')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    plt.plot(history[2], label='train_acc')
    plt.plot(history[3], label='val_acc')
    plt.title('Accuracy trend')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    #Show some misclassified samples    
    if(labels):       
        #get some wrong predictions
        mis_idx = np.where(y != yhat)[0]
        size = min(len(mis_idx), 16)
        wrong_preds = np.random.choice(mis_idx, size=size) 
        ax = []
        fig=plt.figure(figsize=(12, 12))
        columns = 4
        rows = 4
        for i, j in enumerate(wrong_preds):
            if(type(x) is np.ndarray):
                img = x[j]
            else:
                img = plt.imread(x[j])
            ax.append(fig.add_subplot(rows, columns, i+1))
            ax[-1].set_title(f'true: {labels[y[j]]}, pred: {labels[yhat[j]]}')                             
            plt.imshow(img)
        plt.tight_layout(pad=1.2)    
        fig.suptitle('Wrong predictions', y = 0.001)
        plt.show()
    
def print_model_metrics(y, yhat, labels, title, stream=sys.stdout, wrong_preds=False):
    #Check if y is one hot encoded
    if(len(y.shape) != 1):
        y = y.argmax(axis=1)
        yhat = yhat.argmax(axis=1)
        
    print('\n' + title + '\n------------\n', file=stream)   
    
    print("Classification Metrics\n----------\n", file=stream)  
    print(met.classification_report(y, yhat, target_names=labels,
                                    zero_division=1), file=stream) 
    
    print("Confusion Matrix\n----------\n", file=stream)
    print(met.confusion_matrix(y, yhat), file=stream)    
        
    if(wrong_preds):
        print("Wrong Predictions\n----------\n", file=stream)
        mis_idx = np.where(y != yhat)[0]
        size = min(len(mis_idx), 10)
        wrong_preds = np.random.choice(mis_idx, size=size)
        for i in range(size):
            #print(df_test.iloc[wrong_preds[i]], file=stream)
            print('Original Label : {0}'.format(y[wrong_preds[i]]), 
                  file=stream)
            print('Predicted Label : {0}'.format(yhat[wrong_preds[i]]),
                  file=stream)
            print('********************', file=stream)

    
pt.manual_seed(23)
batchsize = 40

#Single gpu train
gpu = pt.device("cuda:0")



pre_proc = tv.transforms.Compose([tv.transforms.Resize((224,224)),
                                  tv.transforms.RandomAffine(0, scale=(0.8,1.2), 
                                                             shear=10),
                                  tv.transforms.RandomHorizontalFlip(),
                                  tv.transforms.ToTensor(),
                                  tv.transforms.Normalize((0.5, 0.5, 0.5), 
                                                          (0.5, 0.5, 0.5))])  

data = tv.datasets.ImageFolder('../Data/101_ObjectCategories', transform=pre_proc)
train_size = int(0.7 * len(data))
val_size = len(data) - train_size
train_data, test_data = pt.utils.data.random_split(data, [train_size, val_size])

train_loader = pt.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, 
                                        pin_memory=True)
test_loader = pt.utils.data.DataLoader(test_data, batch_size=batchsize, pin_memory=True)
class_labels = data.classes

model = tv.models.resnet50(pretrained=True)

for layers in list(model.children())[:-1]:
    for param in layers.parameters():
        param.requires_grad = False

final_layer = pt.nn.Linear(model.fc.in_features, len(class_labels))
model.fc = final_layer
model = pt.nn.DataParallel(model, device_ids=[0, 1])
model.to(gpu)
learning_rate = 0.1
optimizer = pt.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95, 
                         nesterov=True)
#Step based decay
scheduler = pt.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)
objective = pt.nn.CrossEntropyLoss()
epochs = 50
tloss_history=[]
vloss_history=[]
tacc_history=[]
vacc_history=[]

for i in range(epochs):   
    #train on batch
    tloss, tacc = train(model, gpu, train_loader, objective, optimizer, i+1, 25)
    vloss, vacc = test(model, gpu, test_loader, objective)
    #Decay learning rate
    scheduler.step()
    #keep track of losses
    tloss_history.append(tloss)
    vloss_history.append(vloss)
    tacc_history.append(tacc)
    vacc_history.append(vacc)
    print(f'Epoch : {i+1} , Learning Rate : {scheduler.get_lr()}')
    print(f'Epoch : {i+1} => train loss={tloss:.4f}, train acc={tacc:.4f}')
    print(f'............ val loss={vloss:.4f}, val acc={vacc:.4f}')
    #checkpoint code
    if i > 15 :
        if tacc - vacc < .02 and vacc > 0.6:
            pt.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': vloss,
            }, f=f'../Models/epoch{i}_vacc{vacc:.4f}.pt')

#Final predictions
y_test = np.zeros(len(test_data), dtype=np.int32)
y_pred = np.zeros(len(test_data), dtype=np.int32)
with pt.no_grad():
    for i, (data,labels) in enumerate(test_loader):
        data, labels = data.to(gpu), labels.to(gpu)
        samples = len(data)        
        output = model(data)
        preds = predict(output).flatten()
        #Copy the data from gpu memory to cpu before passing to numpy
        preds_numpy = preds.cpu().numpy()
        y_pred[i*samples:(i*samples)+samples] = preds_numpy
        y_test[i*samples:(i*samples)+samples] = labels.cpu().numpy()

#plot loss curves and missed predictions
x_test = [x[0] for x in test_data.dataset.imgs]
model_history = [tloss_history, vloss_history, tacc_history, vacc_history]
plot_loss_wrong_preds(model_history, x_test, y_test, y_pred, class_labels)

#save metrics
file_stream = open('results/caltech101.txt', 'w')
print("Model Summary\n----------\n", file=file_stream)  
#print(summary.summary(model, pt.zeros((1, 3, 224, 224), device=gpu)), file=file_stream)
#print_model_metrics(y_train, y_pred_train, class_labels, 'Train Metrics', file_stream)
print_model_metrics(y_test, y_pred, class_labels, 'Test Metrics', file_stream)
file_stream.close()
#print_model_metrics(y_test, y_pred, class_labels, 'Test Metrics')

#save the model