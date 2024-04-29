import torch
import torch.nn as nn #parent object of nn models
import torch.nn.functional as F #activation function
from torch.utils.data import Dataset,DataLoader

#how we will feed in the training data 
"""NLP techiques 
1. first tokenize the sentence which will split a sentence into individual units such as words and punctuation
2. lower the words so not capitalize and apply stemming which is technique that aims to crudely chop off ends of words like organizer and organizes is organ
3. then we calculate bag of words which is put 0 and 1 for each word in the sentence in pre existing bag of words """
from turtle import forward


class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes) -> None:
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,num_classes)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        #w
        return out