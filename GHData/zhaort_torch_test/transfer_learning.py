import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCH = 1

model = models.vgg19(pretrained=False)
pre = torch.load(r'D:\BaiduNetdiskDownload\models\vgg19-dcbb9e9d.pth')
model.load_state_dict(pre)
print(model)

for param in model.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()


def load_train_data(path):
    path = path
    train_data = datasets.ImageFolder(path,
                                      transform=transforms.Compose([
                                          transforms.Resize((500, 500)),
                                          transforms.ToTensor()])
                                      )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE,
                                   shuffle=True)
    return train_loader


def load_test_data(path):
    path = path
    test_data = datasets.ImageFolder(path,
                                     transform=transforms.Compose([
                                         transforms.Resize((500, 500)),
                                         transforms.ToTensor()])
                                     )
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE,
                                  shuffle=True)
    return test_loader

# train_loader = load_train_data('C:\\Users\\45569\\Pictures\\Camera' + ' ' + 'Roll\\train')
model.load_state_dict(torch.load('net_params.pkl'))
test_loader = load_test_data('C:\\Users\\45569\\Pictures\\Camera' + ' ' + 'Roll\\test')
for step, (x, y) in enumerate(test_loader):
    output = model(x)
    print(output)
# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader):
#         output = model(x)
#         loss = loss_fn(output, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if step % 50 == 0:
#             print(step, loss.item())
#
# torch.save(model.state_dict(), 'net_params.pkl')
