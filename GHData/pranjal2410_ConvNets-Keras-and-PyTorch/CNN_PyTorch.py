import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import cv2
import os
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

train_set_path = 'dataset/training_set/'
test_set_path = 'dataset/test_set/'

train_cats = os.listdir('dataset/training_set/cats/')
train_dogs = os.listdir('dataset/training_set/dogs/')

test_cats = os.listdir('dataset/test_set/cats/')
test_dogs = os.listdir('dataset/test_set/dogs/')

train_images = []
train_labels = []
test_images = []
test_labels = []

for i in train_cats:
    image = cv2.imread(train_set_path + 'cats/' + i)
    img = cv2.resize(image, (64, 64))
    train_images.append(img)
    train_labels.append(0)

for i in train_dogs:
    image = cv2.imread(train_set_path + 'dogs/' + i)
    img = cv2.resize(image, (64, 64))
    train_images.append(img)
    train_labels.append(1)

for i in test_cats:
    image = cv2.imread(test_set_path + 'cats/' + i)
    img = cv2.resize(image, (64, 64))
    test_images.append(img)
    test_labels.append(0)

for i in test_dogs:
    image = cv2.imread(test_set_path + 'dogs/' + i)
    img = cv2.resize(image, (64, 64))
    test_images.append(img)
    test_labels.append(1)

train_images = np.array(train_images, dtype='float') / 255.0
test_images = np.array(test_images, dtype='float') / 255.0

train_images.reshape(8000, 3, 64, 64)
test_images.reshape(2000, 3, 64, 64)

train_labels = to_categorical(np.array(train_labels), num_classes=2)
test_labels = to_categorical(np.array(test_labels), num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=0)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
test_images = torch.from_numpy(test_images).float()

y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
test_labels = torch.from_numpy(test_labels).float()


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(2304, 64)
        self.linear2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 2304)
        x = F.relu(self.linear1(x))
        return F.softmax(self.linear2(x))

    def num_flat_features(self, x):
        size = x.size[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


classifier = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.01)
if torch.cuda.is_available():
    model = classifier.cuda()
    criterion = criterion.cuda()

print(model)


def train(epoch, X_train, X_test, y_train, y_test):
    model.train()
    tr_loss = 0

    X_train, y_train = torch.autograd.Variable(X_train), torch.autograd.Variable(y_train)

    X_test, y_test = torch.autograd.Variable(X_test), torch.autograd.Variable(y_test)

    if torch.cuda.is_available():
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    optimizer.zero_grad()
    output_train = model(X_train)
    output_val = model(X_test)

    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_test)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        print(f'''Epoch: {epoch+1}\t loss: {loss_val}''')

n_epochs = 25

train_losses = []
val_losses = []

X_train = X_train.reshape((6400, 3, 64, 64))
X_test = X_test.reshape((1600, 3, 64, 64))
test_images = test_images.reshape((2000, 3, 64, 64))

for epoch in range(n_epochs):
    train(epoch, X_train, X_test, y_train, y_test)
