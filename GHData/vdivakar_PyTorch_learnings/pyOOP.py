#class encapsulates attributes and methods
class Lizard:
    def __init__(self, name):
        self.name = name #attribute
        print("Created Lizard object with name: ", name)
    
    def set_name(self, name): #method
        self.name = name
        print("Changed name to: ", name)

lizard = Lizard("Blue Lizard")
print(lizard.name) #accessing class attribute

############################Dummy Net#############
import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = None

    def forward(self, t):
        # t = self.layer(t)
        t = t+100
        return t

net = Network()
print(net)
print(net.forward(torch.rand(3,3)))