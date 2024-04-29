import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms

from data.data_utils import load_train_data, load_test_data, get_dataset_classes
from model import CNN_classifier
from model_utils import test_model, get_lr
from tqdm import tqdm
import torch.nn as nn

from utils import get_args, imshow

parser = argparse.ArgumentParser(description='Visualize Pretrained Models')

args = get_args(parser)
model_size = args.model_size
dataset_name = args.dataset_name
verbose = args.verbose
batch_size = args.batch_size
num_workers = args.num_workers
lr = args.learning_rate
weight_decay = args.weight_decay
epochs = args.epochs

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# dataset loaders
train_loader, classes = load_train_data(dataset_name, batch_size, num_workers)
test_loader, test_dataset = load_test_data(dataset_name, batch_size, num_workers)
num_classes = len(classes)
# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
image_shape = list(images[0].size())
# show images
# imshow(torchvision.utils.make_grid(images))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net = CNN_classifier(device, args, image_shape, num_classes)
net.to(device)
# print(net)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:  # print every 2000 mini-batches
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f} lr:{get_lr(optimizer):.5f}')
    running_loss = 0.0
    test_model(net, test_loader, classes, device)

print('Finished Training')

PATH = f'./models/{dataset_name}_net_{model_size}.pth'
if not os.path.isdir(f'./models/{dataset_name}/'):
    os.mkdir(f'./models/{dataset_name}/')
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{str(classes[labels[j]]):5s}' for j in range(4)))

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{str(classes[predicted[j]]):5s}'
                              for j in range(4)))

test_model(net, test_loader, classes, device)

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {str(classname):5s} is {accuracy:.1f} %')
