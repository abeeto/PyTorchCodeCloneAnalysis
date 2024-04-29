#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
from matplotlib import pyplot as plt
import cnn

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
# %%
lr = 1e-3
batch_size = 64
num_epochs = 5
# %%
model = cnn.CNN().to(device)
print(model)
# %%
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# %%
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.ToTensor()
)

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=False)
# %%
model.train()
for epoch in range(num_epochs):  # loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predictions = model(images)
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))
        
print('Finished Training')
# %%
PATH = '/Users/sean/Documents/Neubauer_Research/MNIST/cnn_model'
torch.save(model.state_dict(), PATH)
# %%
correct = 0
total = 0
classes = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        predictions = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predictions = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.2f} %')
# %%
incorrect_examples = []
incorrect_labels = []
incorrect_pred = []
model.eval()
for data, target in test_loader:

    data , target = data.to(device), target.to(device)
    predictions = model(data) # shape = torch.Size([batch_size, 10])
    prediction = predictions.argmax(dim=1, keepdim=True) # prediction will be a 2d tensor of shape [batch_size, 1]
    idxs_mask = ((prediction == target.view_as(prediction))==False).view(-1)
    if idxs_mask.numel(): # if index masks is non-empty append the correspoding data value in incorrect examples
        incorrect_examples.append(data[idxs_mask].squeeze().cpu().numpy())
        incorrect_labels.append(target[idxs_mask].cpu().numpy()) # the corresponding target to the misclassified image
        incorrect_pred.append(prediction[idxs_mask].squeeze().cpu().numpy()) # the corresponding predicted class of the misclassified image
# %%
plt.imshow(incorrect_examples[5])
plt.show()
print(incorrect_labels[5], incorrect_pred[5])
# %%
plt.imshow(incorrect_examples[103])
plt.show()
print(incorrect_labels[103], incorrect_pred[103])
# %%
plt.imshow(incorrect_examples[103][1])
plt.show()
print(incorrect_labels[103][1], incorrect_pred[103][1])
