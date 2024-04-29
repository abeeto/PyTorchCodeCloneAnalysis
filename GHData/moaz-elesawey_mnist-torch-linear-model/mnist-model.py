import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


class Net(nn.Module):
    def __init__(self, n_classes=10, batch_size=16, n_epochs=5, lr=0.001):
        super(Net, self).__init__()
        
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.acc_history = []
        self.loss_history = []
        
        self.train_dl, self.test_dl = self.get_data()
        
        self.fc1 = nn.Linear(28*28, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, n_classes)
        
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        
        return out
    
    def get_data(self):
        train_ds = MNIST(root='./mnist', download=True, train=True, 
                         transform=transforms.ToTensor())
        
        test_ds = MNIST(root='./mnist', download=True, train=False, 
                         transform=transforms.ToTensor())
        
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size)
        
        return train_dl, test_dl
    
    def _train(self):
        self.train()
        
        for epoch in range(self.n_epochs):  
            ep_loss = 0
            ep_acc = []
            for i, (images, labels) in enumerate(self.train_dl):
                images = images.view([-1, 28*28])
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # zero the gradiant
                self.optimizer.zero_grad()
                
                # forward path
                predictions = self.forward(images)
                loss = self.criterion(predictions, labels)
                
                predictions = F.softmax(predictions, dim=1)
                classes = T.argmax(predictions, dim=1)
                wrong = T.where(classes != labels,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size
                ep_acc.append(acc.item())
                ep_loss += loss.item()
                
                self.acc_history.append(acc.item())
                self.loss_history.append(loss.item())
                
                # backward path
                loss.backward()
                self.optimizer.step()
                
            # metrics
            print(f"Epoch [{epoch+1}/{self.n_epochs}] loss: [{ep_loss:.4f}], acc: [{np.mean(ep_acc):.4f}]")
    
    def _test(self):
        self.eval()
        with T.no_grad():
            ep_acc = []
            ep_loss = 0
            for i, (images, labels) in enumerate(self.test_dl):
                images = images.view([-1, 28*28])
                images = images.to(self.device)
                labels = labels.to(self.device)
        
                # forward path
                predictions = self.forward(images)
                loss = self.criterion(predictions, labels)
                
                predictions = F.softmax(predictions, dim=1)
                classes = T.argmax(predictions, dim=1)
                wrong = T.where(classes != labels,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size
                ep_acc.append(acc.item())
                ep_loss += loss.item()
                
            print(f"loss: [{ep_loss:.4f}], acc: [{np.mean(ep_acc):.4f}]")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', '-bs', type=int, default=32,
                        help="enter your batch size")
    parser.add_argument('--epochs', '-ep', type=int, default=5,
                        help="enter your number of epochs") 
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001,
                        help="enter your learning rate")
    parser.add_argument('--save', type=bool, default=False,
                        help="check if you want to save the model.")

    args = parser.parse_args()
    sys.stdout.write(str(run(args)))

def run(args):
    net = Net(batch_size=args.batch_size,
              n_epochs=args.epochs,
              lr=args.learning_rate)
    print("....Start training....")
    net._train()
    print("....Start testing....")
    net._test()

    if args.save:
        T.save(net.state_dict(), "./model")
    
    plt.plot(net.acc_history)
    plt.title("Accuracy plot")
    plt.xlabel("epochs range")
    plt.ylabel("Accuracy")
    plt.show()
    
    plt.plot(net.loss_history)
    plt.title("Loss plot")
    plt.xlabel("epochs range")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()