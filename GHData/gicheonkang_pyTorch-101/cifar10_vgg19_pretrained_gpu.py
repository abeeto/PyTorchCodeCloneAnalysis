import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.models.vgg import model_urls
import torch.nn.functional as F
from torch.autograd import Variable

#hyper parameters
data_dir = './root'
batch_size = 100
num_epoch = 10
learning_rate = 0.01

train_dataset = dsets.CIFAR10(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

test_dataset = dsets.CIFAR10(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
vgg19 = torchvision.models.vgg19_bn(pretrained=True)
use_gpu = torch.cuda.is_available()

'''
if use_gpu:
    vgg19.cuda(0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg19.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        image = Variable(images)
        label = Variable(labels)

        print(image.size())
        print(label.size())
        break
'''























