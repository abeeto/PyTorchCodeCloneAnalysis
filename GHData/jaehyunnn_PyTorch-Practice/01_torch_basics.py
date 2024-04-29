import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

### autograd example 1 ###

# Create tensors
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph
y = w*x + b

# Compute gradient
y.backward()

# Print out the gradients
print(x.grad)
print(w.grad)
print(b.grad)


### autograd example 2 ###

# Create tensors of shape (10, 3) and (10, 2)
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('w: ', linear.bias)

# Build loss function and optimizer
cost = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Foward pass
pred = linear(x)

# Compute loss
loss = cost(pred, y)
print('loss: ', loss.item())

# Backward pass
loss.backward()

# Print out the gradients
print('dL/dw: ', linear.weight.grad)
print('dL/db ', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

# Print out the loss after 1-step gradient descent
pred = linear(x)
loss = cost(pred, y)
print('loss after 1 step optimization: ', loss.item())


### Loading data from numpy ###

# Create a numpy array
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array
z = y.numpy()


### Input pipline ###

# Download and construc CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

# Fetch one data pair (read data from disk)
image, label = train_dataset[0]
print(image.size())
print(label)

# Data loader (this provides queues and threads in a very simple way)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files
data_iter = iter(train_loader)

# Mini-batch images and labels
images, labels = data_iter.next()

# Actual usage of the data loader is as below
for images, labels in train_loader:
    pass


### Input pipline for custom dataset ###

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        """
        TODO
        1. Initialize file paths or a list of file names.
        """
    def __getitem__(self, index):
        """
        TODO
        1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open)
        2. Preprocess the data (e.g. torchvision.Transform)
        3. Return a data pair (e.g. image and label)
        """
        pass
    def __len__(self):
        # Change - to the total size of dataset
        return 0

custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)


### Pretrained model ###

# Download and load the pretrained ResNet-18
resnet = torchvision.models.resnet18(pretrained=True)

# If want to finetune only the top layer of the model, set as below
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# Forward pass
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())


### Save and load the model ###

# Save and load the entire model
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended)
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))



# ================================================================== #
#                           [References]                             #
#                                                                    #
#           https://github.com/yunjey/pytorch-tutorial.git           #
# ================================================================== #