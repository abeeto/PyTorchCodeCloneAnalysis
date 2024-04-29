import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=100)
        self.fc2 = nn.Linear(in_features=100,out_features=3)
        self.out = nn.Linear(in_features=3,  out_features=1)

    # def __repr__(self):
    #     #To override default string representation of class object
    #     return "This is my basic network"

    def forward(self, input):
        #To be implemented
        return input

net = Network()
print(net)
print(net.fc1)
print(net.fc1.weight.shape)

print('*'*10, "printing all parameters:")
for param in net.parameters():
    print(param.shape)

print("*"*10, "printing with names:")
for name, param in net.named_parameters():
    print(name, "\t", param.shape)

print("-"*20)
my_mat = torch.tensor([
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4]
])
vec = torch.tensor([1,2,3,4])
print(my_mat.matmul(vec))