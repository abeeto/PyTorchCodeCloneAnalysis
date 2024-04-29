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

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCH = 2


class MyDataset(Data.Dataset):
    def __init__(self, file_path, transform=None, ):
        super(MyDataset, self).__init__()
        images = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                words = line.split()
                images.append((words[0], int(words[1])))

        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.images[index]
        img = Image.open(file_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


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


train_loader = load_train_data('C:\\Users\\45569\\Pictures\\Camera' + ' ' + 'Roll\\train')


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(288, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()
cnn.load_state_dict(torch.load('net_params.pkl'))
test_loader = load_test_data('C:\\Users\\45569\\Pictures\\Camera' + ' ' + 'Roll\\test')
for step, (x, y) in enumerate(test_loader):
    output = cnn(x)
    print(output)

# optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
# loss_fn = nn.CrossEntropyLoss()
#
#
# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader):
#         output = cnn(x)
#         loss = loss_fn(output, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if step % 50 == 0:
#             print(step, loss.item())
#
# torch.save(cnn.state_dict(), 'net_params.pkl')
