import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # Set this flag to set your devices. For example if I set '6,7', then cuda:0 and cuda:1 in code will be cuda:6 and cuda:7 on hardware


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)  # Reshapes image to 1-D tensor
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.softmax(self.layer_3(x))
        return x


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    trainloader = DataLoader(mnist_train, batch_size=60000, num_workers=2, shuffle=True)    # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
    testloader = DataLoader(mnist_test, batch_size=10000, num_workers=2, shuffle=True)      # Code will run so much faster without dataloaders for small(ish) datasets
    model = Model()
    no_epochs = 10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to('cuda:0')
    train_loss = []
    test_loss = []
    for epoch in range(no_epochs):
        model.train()
        total_loss = 0
        for itr, (x, y) in enumerate(trainloader):
            x, y = x.to('cuda:0'), y.to('cuda:0')
            optimizer.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Epoch {}/{}: Training Loss: {:4f}'.format(epoch+1, no_epochs, total_loss))
        train_loss.append(total_loss)
        model.eval()    # This removes stuff like dropout and batch norm for inference stuff
        total_loss = 0
        for itr, (x, y) in enumerate(testloader):
            x, y = x.to('cuda:0'), y.to('cuda:0')
            outputs = model.forward(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
        print('Test Loss: {:4f}'.format(total_loss))
        test_loss.append(total_loss)
    plt.plot(np.arange(no_epochs), train_loss, label='Train Loss')
    plt.plot(np.arange(no_epochs), test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # torch.save(model.state_dict(), os.getcwd()+'/saved_models/mlp.pt')    # To save model parameters, uncomment
    # model.load_state_dict(torch.load(os.getcwd()+'/saved_models/d.pt'))   # Use this to load them back in (obviously somewhere else)


if __name__ == '__main__':
    main()
