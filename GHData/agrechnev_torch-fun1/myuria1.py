import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision

# My simple SGD optimizer
class MySGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super(MySGD, self).__init__(params, dict(lr=lr))

    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError('closure')
        for group in self.param_groups:
            # print('lr =', group['lr'])
            for p in group['params']:
                p.data.add_(-group['lr'], p.grad.data)

########################################################################################################################

class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 10, 3, 1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(10, 10, 3, 1, padding=1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.conv2_1 = torch.nn.Conv2d(10, 20, 3, 1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(20, 20, 3, 1, padding=1)
        self.dropout2 = torch.nn.Dropout2d(0.25)
        self.conv3_1 = torch.nn.Conv2d(20, 40, 3, 1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(40, 40, 3, 1, padding=1)
        self.dropout3 = torch.nn.Dropout2d(0.25)
        self.fc1 = torch.nn.Linear(640, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1_1(x))
        x = torch.nn.functional.relu(self.conv1_2(x))
        x = self.dropout1(x)
        #
        x = torch.nn.functional.max_pool2d(x, 2)
        #
        x = torch.nn.functional.relu(self.conv2_1(x))
        x = torch.nn.functional.relu(self.conv2_2(x))
        x = self.dropout2(x)
        #
        x = torch.nn.functional.max_pool2d(x, 2)
        #
        x = torch.nn.functional.relu(self.conv3_1(x))
        x = torch.nn.functional.relu(self.conv3_2(x))
        x = self.dropout3(x)
        #
        x = torch.nn.functional.max_pool2d(x, 2)
        #
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        #
        return x


########################################################################################################################

class Trainer:
    ########
    def __init__(self):
        self.device = 'cuda'
        self.n_classes = 10
        self.n_epoch = 100
        self.batch_size = 64
        self.lr = 0.01
        self.class_names = ['plane', 'car', 'bird', 'cat',
                            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Load cifar-10
        tran = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.set_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tran)
        self.set_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tran)

        print('len(set_train) =', len(self.set_train))
        print('set_train =', self.set_train)
        print('len(set_test) =', len(self.set_test))
        print('set_test =', self.set_test)

        if False:
            plt.figure(figsize=(18, 11))
            nx, ny = 9, 7
            for idx in range(nx * ny):
                plt.subplot(ny, nx, idx + 1)
                x, y = self.set_train[np.random.randint(len(self.set_train))]
                # print(x.shape, x.dtype, x.min(), x.max())
                # sys.exit(0)
                x = np.array(x).transpose(1, 2, 0)
                x = (x + 1) / 2
                plt.imshow(x)
                plt.axis('off')
                plt.title(self.class_names[y])
            plt.show()

        # Create the loaders
        self.loader_train = torch.utils.data.DataLoader(self.set_train, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=2)
        self.loader_test = torch.utils.data.DataLoader(self.set_test, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=2)

        # The net
        self.net = Net1().to(device=self.device)
        print('net =', self.net)

        # Criterion + optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.net.parameters())
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        # self.optimizer = MySGD(self.net.parameters(), lr=self.lr)

    ########
    def train(self):
        self.net.train()
        print('len(self.loader_train) =', len(self.loader_train))
        for epoch in range(self.n_epoch):
            loss_sum = 0.
            print(f'Epoch {epoch} of {self.n_epoch}')
            for i, (batch_x, batch_y) in enumerate(self.loader_train):
                # Train 1 batch
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                out = self.net(batch_x)
                loss = self.criterion(out, batch_y)
                # loss = torch.nn.functional.nll_loss(out, batch_y)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
                # print('loss = ', loss.item())
                # print('grad = ', self.net.fc2.weight.grad)
                # sys.exit(0)

            print('avg_loss = ', loss_sum / len(self.loader_train.dataset))
def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
