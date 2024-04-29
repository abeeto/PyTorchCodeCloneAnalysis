import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import optim

BATCH_SIZE = 50

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

N_STEPS = 28
N_INPUTS = 28
N_NEURONS = 150
N_OUTPUTS = 10
N_EPHOCS = 10


class ImageRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageRNN, self).__init__()

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs

        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons, batch_first=True)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(1, self.batch_size, self.n_neurons)

    def forward(self, x):
        self.hidden = self.init_hidden()

        rnn_out, self.hidden = self.basic_rnn(x, self.hidden)
        out = self.FC(self.hidden)          # moore Machine

        return out.view(-1, self.n_outputs)


#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#logits = model(images.view(-1, 28, 28))

model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cpu')


def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()


def run():
    for epoch in range(N_EPHOCS):
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, data in enumerate(trainloader, 1):
            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            input_im, labels = data
            inputs = input_im.view(-1, 28, 28)
            outputs = model(inputs)

            print(f"{outputs=}\n{outputs.size()}\n{labels=}\n{labels.size()}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, labels, BATCH_SIZE)
            print(f"EPOCH: {epoch} | LOSS: {train_running_loss / i:.4f} | ACCURACY: {train_acc / i}", end='\n')
        model.eval()
        print()


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
