import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import numpy as np
import torch.utils.data as data_utils
import pandas as pd


path_to_training_data = '|| PASTE YOUR PATH FOR TRAINING HERE ||'
path_to_labels_data = '|| PASTE YOUR PATH FOR LABELS HERE ||'

X = pd.read_pickle(path_to_training_data)
Y = pd.read_pickle(path_to_labels_data)

X = np.asarray(X).astype('float64')
Y = np.asarray(Y).astype('float64')

X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
X = Variable(X)
Y = Variable(Y)

if torch.cuda.is_available():
    X = X.cuda()
    Y = Y.cuda()

train = data_utils.TensorDataset(X, Y)
train_loader = data_utils.DataLoader(train, batch_size=128, shuffle=True)

batch_size = 128
epochs = 10

MODEL_STORE_PATH = 'model/'

if not os.path.exists(MODEL_STORE_PATH):
    os.makedirs(MODEL_STORE_PATH)


input_shape = X.shape[1:]

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
#        self.batchNorm = nn.modules.BatchNorm2d(input_shape)
        self.Convlayer = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.Deconvlayer = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(True), nn.Dropout(0.2),
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(True), nn.Dropout(0.2),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(True), nn.Dropout(0.2),
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(True), nn.Dropout(0.2),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=0),
                nn.ReLU(True), nn.Dropout(0.2),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=0),
                nn.ReLU(True))
        
        
    def forward(self, x):
        out = self.Convlayer(x)
        out = self.Deconvlayer(out)
        
        return out

model = SegNet()

if torch.cuda.is_available():
    model.cuda()
else:
    print('Please switch to GPU for faster processing!!!')
    model = model.cpu()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

total_step = len(train_loader)
loss_list = []
acc_list = []

    
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = torch.reshape(images, [len(images), 3, 80, 160])
        images = images.float()
        labels = torch.reshape(labels, [len(labels), 1, 80, 160])
        labels = labels.float()
#         Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

#         Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.long()).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item())

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = torch.reshape(images, [len(images), 3, 80, 160])
        images = images.float()
#        labels = torch.reshape(labels, [len(labels), 1, 80, 160])
        labels = labels.float()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

    print('Test Accuracy of the model : {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'seg_net_model.pth')
