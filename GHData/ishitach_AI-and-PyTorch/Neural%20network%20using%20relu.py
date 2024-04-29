#Creating a neural network with 128,64 hidden layers and 10 output layers and ReLU activation

import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)
        
        
    def forward(self, x):
        # Hidden layer with relu function
        x = self.hidden1(x)
        x= relu(x)
        x = self.hidden2(x)
        x= relu(x)
        x = F.softmax(self.output(x), dim=1)
        
        return x
model = Network()
model
