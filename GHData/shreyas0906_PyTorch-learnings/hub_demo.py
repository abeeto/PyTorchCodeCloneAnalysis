import torch
from torch import hub

# resnet_18_model = hub.load('pytorch/vision:main', 'resnet18', pre_trained=False)
a = torch.tensor([1,2,3,4,5,6], names=['sample'])
print(a)