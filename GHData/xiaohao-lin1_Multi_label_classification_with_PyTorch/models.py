import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torch.optim as optim

#code: https://www.kaggle.com/pmigdal/transfer-learning-with-resnet-50-in-pytorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

model = models.resnet50(pretrained=True)




#freeze the parameters
for parameter in model.parameters():
    parameter.requires_grad = False


#build custom classifier
#TODO: resnet has no classifer, google what is image classification w resnet, check one of abhi's message
#resnet   (fc): Linear(in_features=2048, out_features=1000, bias=True)
fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(128, 6)
)

model.fc = fc

model.to(device)

criterion = nn.CrossEntropyLoss
optimizer = optim.Adam(model.fc.parameters())





