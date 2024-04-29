import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, numClasses)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float() # bug fix? Works on windows? linux or mac might need to delete this line*
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out