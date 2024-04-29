import torch
from torch import  nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import fc_model

transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.FashionMNIST('FashionMNIST_data/', download=False, train=True, transform=transform)
test_set = datasets.FashionMNIST('FashionMNIST_data/', download=False, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

image, label = next(iter(train_loader))

model = fc_model.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# fc_model.train(model, train_loader, test_loader, criterion, optimizer, epochs=2)
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())