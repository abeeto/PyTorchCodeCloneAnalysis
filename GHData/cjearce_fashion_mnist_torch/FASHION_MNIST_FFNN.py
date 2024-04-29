#!/usr/bin/env python
# coding: utf-8

# # Training Deep Neural Networks with PyTorch

# In[1]:


import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
get_ipython().run_line_magic('matplotlib', 'inline')

matplotlib.rcParams['figure.facecolor'] = '#ffffff'


# We can download the data and create a PyTorch dataset using the `MNIST` class from `torchvision.datasets`. 

# In[2]:


#Preparing the Data
dataset = FashionMNIST(root='data/', download=True, transform=ToTensor())
test_dataset = FashionMNIST(root='data/', train=False, transform=ToTensor())


# In[3]:


image, label = dataset[0]
print('image.shape:', image.shape)
plt.imshow(image.permute(1, 2, 0), cmap='gray')
print('Label:', label)
print('Label:', dataset.classes[label])


# In[4]:


image, label = dataset[5]
print('image.shape:', image.shape)
plt.imshow(image.permute(1, 2, 0), cmap='gray')
print('Label:', label)
print('Label:', dataset.classes[label])


# In[5]:


#Using `random_split` helper function, set aside 10000 images as validation set. 
val_size = 10000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)


# We can now create PyTorch data loaders for training and validation.

# In[6]:


batch_size=128


# In[7]:


train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)


# In[8]:


#visualize a batch of data
for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break #1 batch only - break


# ## Model
# 
# Visual representation of the model with only 1 hidden layer (but I will use 3 hidden layers in the code).
# 
# <img src="https://i.imgur.com/eN7FrpF.png" width="480">
# 
# 
# Let's define the model by extending the `nn.Module` class from PyTorch.

# Then, use the Rectified Linear Unit (ReLU) function as the activation function for the outputs. It has the formula `relu(x) = max(0,x)` i.e. it simply replaces negative values in a given tensor with the value 0. ReLU is a non-linear function, as seen here visually:
# 
# <img src="https://i.imgur.com/yijV4xF.png" width="420">
# 
# We can use the `F.relu` method to apply ReLU to the elements of a tensor.

# In[29]:


class MnistModel(nn.Module):
    """Feedfoward neural network with 3 hidden layers"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, 128)
        # hidden layer2
        self.linear2 = nn.Linear(128, 64)
        # hidden layer3
        self.linear3 = nn.Linear(64, 32)
        #output layer
        self.linear4 = nn.Linear(32, out_size)
          
        
    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer 2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer 3
        out = self.linear3(out)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear4(out)
        return out
    
    def training_step(self, batch):
        '''Returns the loss for a batch of training data'''
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
       
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        '''Takes result from the batches and averages the losses and accuracy to get overall'''
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


# In[30]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[31]:


input_size = 784
hidden_size = 32 
num_classes = 10


# In[32]:


model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)


# In[34]:


for images, labels in train_loader:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss:', loss.item())
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)


# ## Training the Model

# In[41]:


def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch) 
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader) 
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[42]:


model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)


# In[43]:


history = [evaluate(model, val_loader)]
history


# In[44]:


history += fit(5, 0.5, model, train_loader, val_loader)


# In[45]:


history += fit(5, 0.1, model, train_loader, val_loader)


# In[46]:


history += fit(10, 0.05, model, train_loader, val_loader)


# In[47]:


history += fit(20, 0.1, model, train_loader, val_loader)


# In[48]:


history += fit(50, 0.01, model, train_loader, val_loader)


# In[49]:


# history += fit(30, 0.1, model, train_loader, val_loader)


# In[50]:


# history += fit(30, 0.001, model, train_loader, val_loader)


# In[51]:


losses = [x['val_loss'] for x in history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs');


# In[52]:


accuracies = [x['val_acc'] for x in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


# ## Testing with individual images

# In[53]:


# Define test dataset
test_dataset = FashionMNIST(root='data/', 
                     train=False,
                     transform=ToTensor())


# Define a helper function `predict_image`, which returns the predicted label for a single image tensor.

# In[54]:


def predict_image(img, model):
    yb = model(img)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


# In[55]:


img, label = test_dataset[5]
plt.imshow(img[0], cmap='gray')
print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# In[56]:


img, label = test_dataset[1839]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# In[57]:


img, label = test_dataset[193]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# In[58]:


img, label = test_dataset[9999]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# In[59]:


img, label = test_dataset[67]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# In[60]:


img, label = test_dataset[48]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))
print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])

