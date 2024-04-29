import torch
import torch.nn as nn 

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classses):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() #activation function
        self.linear2 = nn.Linear(hidden_size, num_classes) #output is the number of classes to be classified into

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #no Softmax at the very end

        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()