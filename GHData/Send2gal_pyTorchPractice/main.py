import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from time import sleep

PATH = None  # r"C:\Users\gcohen\PycharmProjects\pytorchTutorials\models.mod"
EPOCHS = 3
BATCH_SIZE = 10
NAME = 'TEST_1'


train = datasets.MNIST("", train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def train(net, optimizer, writer):
    if not PATH:  # not os.path.isfile(PATH):
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_set.dataset.train_data), BATCH_SIZE)):
                # for data in train_set:
                # x, y = data
                x = train_set.dataset.train_data[i:i + BATCH_SIZE]/255.0
                y = train_set.dataset.train_labels[i:i + BATCH_SIZE]
                net.zero_grad()
                output = net(x.view(-1, 28 * 28))
                loss = F.nll_loss(output, y)
                loss.backward()
                optimizer.step()
                # have to chnage the n_iter to real parmter
                writer.add_scalar('Loss/train', loss , i)
                # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)

                # # print
                # plt.imshow(x[0].view(28, 28))
                # plt.show()

                # print(loss)
        # torch.save(net.state_dict(), PATH)
    else:
        net.load_state_dict(torch.load(PATH))
        net.eval()


def testing(net, writer):
    correct = 0
    loss = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_set.dataset.train_data), BATCH_SIZE)):
            x = train_set.dataset.train_data[i:i + BATCH_SIZE]/255.0
            y = train_set.dataset.train_labels[i:i + BATCH_SIZE]
            output = net(x.view(-1, 28 * 28))
            for idx, i in enumerate(output):
                print(total, int(y[idx]), torch.argmax(i) == y[idx])
                if not torch.argmax(i) == y[idx]:
                    print("ai think ", torch.argmax(i), "real value ", y[idx])
                    loss += 1
                    writer.add_scalar('Loss/test', loss/total+1, i)
                    # plt.imshow(x[idx].view(28, 28))
                    # plt.show()
                    # sleep(3)
                if torch.argmax(i) == y[idx]:
                    correct += 1
                    writer.add_scalar('Accuracy/test', correct/total+1, i)
                total += 1

    print("Accuracy: ", round(correct / total, 3), "from: ", total)


def main():
    net = Net()
    name = NAME + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M')
    writer = SummaryWriter(os.path.join(os.getcwd(), 'runs', name))
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(net, optimizer, writer)
    testing(net, writer)


if __name__ == "__main__":
    main()
