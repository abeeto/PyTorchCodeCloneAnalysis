import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(385, 128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p = 0.15)

        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p = 0.15)

        self.fc_output = nn.Linear(64, 2)


    def forward(self, X, z):
        X = self.maxpool1(self.relu1(self.cnn1(X)))
        X = self.maxpool2(self.relu2(self.cnn2(X)))
        X = self.maxpool3(self.relu3(self.cnn3(X)))
        # This is where we add our speed variable 'z' to our model after all the convolutions
        X = X.view(X.size(0), -1)
        X = torch.cat((X, z), 1)
        X = self.dropout1(self.relu4(self.fc1(X)))
        X = self.dropout2(self.relu5(self.fc2(X)))
        X = self.fc_output(X).view(-1,1,2)
        return X
