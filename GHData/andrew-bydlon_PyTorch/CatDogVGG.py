import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import time
from apex import amp


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp,cmap)


is_cuda = False
if torch.cuda.is_available():
    is_cuda = True
    AMP = input("Would you like to use AMP? [y/n]")
    if AMP == 'y':
        AMP = True

simple_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

datadir = '/home/andrew/PycharmProjects/DeepLearning/CatDog/CatDogWorking/'

train = ImageFolder(os.path.join(datadir, 'train'), simple_transform)
valid = ImageFolder(os.path.join(datadir, 'val'), simple_transform)

train_data_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=32, num_workers=15)
valid_data_loader = torch.utils.data.DataLoader(valid, shuffle=True, batch_size=32, num_workers=15)

print(train.class_to_idx)
print(train.classes)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model = Net()
if is_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
if AMP:
    model, optimizer = amp.initialize(model, optimizer)


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, reduction='sum').item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            if AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
            else:
                loss.backward()
                optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy


train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

ST = time.time()
for epoch in range(1,20):
    print("Epoch #", epoch, sep="")
    epoch_loss, epoch_accuracy = fit(epoch,model,train_data_loader, phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,valid_data_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
print("Time 1:", time.time() - ST, "seconds.")

plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()

plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'bo', label='train accuracy')
plt.plot(range(1, len(val_accuracy)+1), val_accuracy, 'r', label='val accuracy')
plt.legend()

plt.show()

# Importing into VGG

vgg = models.vgg13(pretrained=True)
vgg = vgg.cuda()

# Freeze layers for training
for param in vgg.features.parameters():
    param.requires_grad = False

vgg.classifier[6].out_features = 2

optimizer = optim.SGD(vgg.classifier.parameters(), lr=1e-4, momentum=0.5)

if AMP:
    vgg, optimizer = amp.initialize(vgg,optimizer)


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        running_loss += F.cross_entropy(output, target, reduction='sum').item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            if AMP == True:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
            else:
                loss.backward()
                optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100 * running_correct.item() / len(data_loader.dataset)

    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy


train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

ST=time.time()
for epoch in range(1,10):
    print("Epoch:", epoch)
    epoch_loss, epoch_accuracy = fit(epoch, vgg, train_data_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, vgg, valid_data_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
print("Total time:", time.time()-ST, "seconds.")

epochs = range(1,10)
plt.plot(epochs, train_losses, 'bo', label='Training Loss')
plt.plot(epochs, val_losses, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.figure()

plt.show()

plt.plot(epochs, train_accuracy, 'bo', label='Training Acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.figure()

plt.show()



for layer in vgg.classifier.children():
    if(type(layer) == nn.Dropout):
        layer.p = 0.2

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

ST=time.time()
for epoch in range(1,3):
    epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
print("Total time:", time.time()-ST, "seconds.")

# Data Augmentation

train_transform = transforms.Compose([transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train = ImageFolder(os.path.join(datadir, 'train'), train_transform)
valid = ImageFolder(os.path.join(datadir, 'val'), simple_transform)

train_data_loader = DataLoader(train, batch_size=32, num_workers=3, shuffle=True)
valid_data_loader = DataLoader(valid, batch_size=32, num_workers=3, shuffle=True)

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

ST=time.time()
for epoch in range(1,3):
    epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
print("Total time:", time.time()-ST, "seconds.")

# Preconvoluted Features
features = vgg.features

train_data_loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=32, num_workers=15)
valid_data_loader = torch.utils.data.DataLoader(valid, shuffle=False, batch_size=32, num_workers=15)

def preconvoluted(dataset,model):
    conv_features, labels_list = [], []
    for data in dataset:
        inputs,labels = data
        inputs,labels = inputs.cuda(), labels.cuda()
        output = model(inputs)
        conv_features.extend(output.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())

    conv_features = np.concatenate([[feat] for feat in conv_features])
    return (conv_features,labels_list)

conv_feat_train,labels_train = preconvoluted(train_data_loader,features)
conv_feat_val,labels_val = preconvoluted(valid_data_loader,features)

class My_dataset(Dataset):
    def __init__(self,feat,labels):
        self.conv_feat = feat
        self.labels = labels

    def __len__(self):
        return len(self.conv_feat)

    def __getitem__(self,idx):
        return self.conv_feat[idx], self.labels[idx]

train_feat_dataset = My_dataset(conv_feat_train, labels_train)
val_feat_dataset = My_dataset(conv_feat_val, labels_val)

train_feat_loader = DataLoader(train_feat_dataset, batch_size=64, shuffle=True)
val_feat_loader = DataLoader(val_feat_dataset, batch_size=64, shuffle=True)

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

#ST=time.time()
#for epoch in range(1,20):
#    epoch_loss, epoch_accuracy = fit(epoch,vgg.classifier,train_feat_loader,phase='training')
#    val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg.classifier,val_feat_loader,phase='validation')
#    train_losses.append(epoch_loss)
#    train_accuracy.append(epoch_accuracy)
#    val_losses.append(val_epoch_loss)
#    val_accuracy.append(val_epoch_accuracy)
#print("Total time:", time.time()-ST, "seconds.")

# Visualizing intermediate outputs

class LayerActivations():
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

# conv_out = LayerActivations(vgg.features,0)

# o = vgg(img.cuda())
# conv_out.remove()

# act = conv_out.features