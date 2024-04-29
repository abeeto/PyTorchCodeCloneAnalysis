import torchvision
import torch
from torch import nn

from torch.utils.data import DataLoader

# define the device to train the model
device = torch.device("cuda")

train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
# if train_data_size=10 , the length of train datasets is 10
print("the length of train_data_size is : {}".format(train_data_size))
print("the length of test_data_size is : {}".format(test_data_size))

# use DataLoader to load the data
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# build the NN


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# create the nn
tudui = Tudui()
#transfer the model to gpu
tudui = tudui.to(device)

# create the loss func
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# some parameters
total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    print("------the train number {} begin-----".format(i + 1))

    # start training
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        #print("train count:{},Loss :{}".format(total_train_step, loss.item()))
