import torch.nn as nn
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, input_size=1, num_classes=10):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(3136, num_classes),
            nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.model(x)
        return x
