import torch.nn as nn
import torch

#nn module is callable, the instance of any class can act like function when applied to its arguments
l = nn.Linear(2,5)
v = torch.FloatTensor([1,2]) #tensor [1,2], the two inputs
print(l(v)) #function


#Combining layers into a pipeline
s = nn.Sequential(
    nn.Linear(2,5), #input passed here first
    nn.ReLU(), #output of that input to be passed here as input
    nn.Linear(5,20), #output of previous passed here as input... etc
    nn.ReLU(),
    nn.Linear(20,10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1))

print(s)

print(s(torch.FloatTensor([[1,2]])))
