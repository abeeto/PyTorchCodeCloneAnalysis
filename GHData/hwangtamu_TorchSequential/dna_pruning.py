from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from src.LSTM import SequentialMNIST
from src.data import MNIST
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random, csv
from collections import Counter

#loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()
bins = 100
hist_range = [-1,1]
labels = ['EI','IE','N']

class Splice:
    def __init__(self,
                 p, n_class=3):
        self.n_class = n_class
        self.base = {'A':0,'T':1,'G':2,'C':3,'N':4,'D':5,'R':6,'S':7}
        self.result = {'EI':0,'IE':1,'N':2}
        self.x = None
        self.x_raw = []
        self.x_raw_train = None
        self.x_raw_test = None
        self.y = None
        self.path = p
        self.count = None
        self.train = {}
        self.test = {}
        self.split()

    def convert(self):
        """
        convert DNA sequences to numpy arrays
        :param p: csv file path
        :param n_class: number of classes
        :return:
        """
        with open(self.path) as csvfile:
            reader = csv.reader(csvfile)
            d = []
            for row in reader:
                d+=[row]
            if self.n_class==2:
                dd = []
                for i in range(len(d)):
                    if d[i][0] in ['EI','IE','N']:
                        dd+=[d[i]]
                d = dd

            random.seed(0)
            random.shuffle(d)

            self.x = np.zeros((len(d),len(d[0][2].strip()),4))
            self.y = np.zeros((len(d),self.n_class))
            self.count = Counter([x[0] for x in d])
            for i in range(len(d)):
                self.x_raw += [d[i][2].strip()]
                tmp = [self.base[x] for x in d[i][2].strip()]
                for j in range(len(tmp)):
                    if tmp[j]==4:
                        # N: A or G or C or T
                        self.x[i][j][0] = .25
                        self.x[i][j][1] = .25
                        self.x[i][j][2] = .25
                        self.x[i][j][3] = .25
                    elif tmp[j]==5:
                        # D: A or G or T
                        self.x[i][j][0] = .33
                        self.x[i][j][1] = .33
                        self.x[i][j][2] = .33
                    elif tmp[j]==6:
                        # R: A or G
                        self.x[i][j][0] = .50
                        self.x[i][j][2] = .50
                    elif tmp[j]==7:
                        # S: C or G
                        self.x[i][j][2] = .50
                        self.x[i][j][3] = .50
                    else:
                        self.x[i][j][tmp[j]] = 1

                #self.x[i][range(len(tmp)),tmp] = 1
                self.y[i][self.result[d[i][0]]] = 1

    def split(self):
        self.convert()
        self.train['x'], self.test['x'] = self.x[:int(self.x.shape[0] * 0.8)], self.x[int(self.x.shape[0] * 0.8):]
        self.train['y'], self.test['y'] = self.y[:int(self.y.shape[0] * 0.8)], self.y[int(self.y.shape[0] * 0.8):]
        self.x_raw_train, self.x_raw_test = self.x_raw[:int(self.x.shape[0] * 0.8)], self.x_raw[int(self.x.shape[0] * 0.8):]

        self.train['x'] = torch.stack([torch.Tensor(x) for x in self.train['x']])
        self.train['y'] = torch.stack([torch.Tensor(x) for x in self.train['y']])
        self.test['x'] = torch.stack([torch.Tensor(x) for x in self.test['x']])
        self.test['y'] = torch.stack([torch.Tensor(x) for x in self.test['y']])


def get_output(n):
    path = './model/dna/'+str(n)+'_lstm.model'
    splice = Splice('./DNA/splice.data')

    train_ = TensorDataset(splice.train['x'], splice.train['y'])
    test_ = TensorDataset(splice.test['x'], splice.test['y'])
    train_loader = DataLoader(train_, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_,batch_size=32, shuffle=False)
    device = torch.device("cuda")
    x_ = []
    y_ = []
    with torch.no_grad():

        model = SequentialMNIST(64, n).to(device)
        model.load(path)
        w = model.hidden2label.weight.data
        print(list(model.parameters())[-2])
        w = w.cpu().numpy()
        w_2 = np.mean(np.absolute(w), axis=0)
        print(np.argmin(w_2), np.min(w_2))
        print(np.argmax(w_2), np.max(w_2))
        w_sorted = np.argsort(w_2)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model.get_hidden(data, path)
            print(output[0].size())
            x = output[0][-1].cpu().numpy().T
            y = target.cpu().numpy()
            x_ += list(x[w_sorted[-4]])
            print(y.shape)
            y_ += list(np.argmax(y, axis=1))
            # print(x.shape, y.shape)
            # plt.scatter(y,x[np.argmax(w_2)], c='b', alpha=0.1)
        #     corr = np.corrcoef([*x,y])[-1][:-1]
        #     plt.scatter(list(range(256)),corr,c='b',alpha=0.1)
        x_ = np.array(x_)
        y_ = np.array(y_)
        datasets = [x_[y_==t] for t in range(3)]
        binned_data_sets = [
            np.histogram(d, range=hist_range, bins=100)[0]
            for d in datasets
            ]
        binned_maximums = np.max(binned_data_sets, axis=1)
        print(binned_maximums)
        x_locations = np.arange(0, sum(binned_maximums), sum(binned_maximums)//3)
        bin_edges = np.linspace(hist_range[0], hist_range[1], 100 + 1)
        centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
        heights = np.diff(bin_edges)

        # Cycle through and plot each histogram
        fig, ax = plt.subplots()
        for x_loc, binned_data in zip(x_locations, binned_data_sets):
            lefts = x_loc - 0.5 * binned_data
            ax.barh(centers, binned_data, height=heights, left=lefts, color='b')

        ax.set_xticks(x_locations)
        ax.set_xticklabels(labels)

        ax.set_ylabel("Activation")
        ax.set_xlabel("Label")

        plt.grid()
        plt.show()




def lesion_test(n, lesion):
    path = './model/dna/' + str(n) + '_gru_d.model'
    splice = Splice('./DNA/splice.data')

    train_ = TensorDataset(splice.train['x'], splice.train['y'])
    test_ = TensorDataset(splice.test['x'], splice.test['y'])
    train_loader = DataLoader(train_, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_, batch_size=32, shuffle=False)
    device = torch.device("cuda")
    model = SequentialMNIST(64, n).to(device)
    model.load(path)
    w = model.hidden2label.weight.data
    w = w.cpu().numpy()
    w_2 = np.mean(np.absolute(w), axis=0)
    w_sorted = np.argsort(w_2)
    blocked = w_sorted[:256-lesion]
    model2 = SequentialMNIST(64, n, blocked=blocked).to(device)
    model2.load(path, blocked)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target=target.long()
            target = torch.max(target, 1)[1]
            data, target = data.to(device), target.to(device)
            # model.get_hidden(data, path)
            # output = model.show_pred(data, path)
            # for i in range(target.size(0)):
            #     print(target[i].cpu().numpy(), output[1][i].cpu().numpy())
            output = model2(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

if __name__ == '__main__':
    # get_output(256)
    a = []
    for i in [4,6,8,10,12,16,32,64,128,256]:
    #     main(i)
        a+=[lesion_test(256, i)]
    print(sorted(a))