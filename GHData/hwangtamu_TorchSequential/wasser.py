from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from src.LSTM import SequentialMNIST
from src.data import MNIST
from src.LSTMencoder import Autoencoder
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import  summary
from collections import Counter
from scipy.stats import wasserstein_distance
from matplotlib.lines import Line2D

#loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()
bins = 100
hist_range = [-1,1]
labels = [str(x) for x in range(10)]
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import torch.utils.data as utils
import matplotlib.pylab as pl

torch.manual_seed(0)
colors = pl.cm.jet(np.linspace(0,1,10))

legend_elements = [Line2D([0], [0], marker='o', color='w', label=str(i),
                          markerfacecolor=c, markersize=5) for i,c in enumerate(colors)]


def get_output(n,i):
    path = './model/new/'+str(n)+'_lstm.model'
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader
    device = torch.device("cuda")
    x_ = []
    y_ = []
    with torch.no_grad():

        model = SequentialMNIST(64, n).to(device)
        model.load(path)
        # for k in model.state_dict():
        #     print(k, model.state_dict()[k].size())

        w = model.hidden2label.weight.data
        # print(list(model.parameters())[-2])
        w = w.cpu().numpy()
        w_2 = np.mean(np.absolute(w), axis=0)
        # print(np.argmin(w_2), np.min(w_2))
        # print(np.argmax(w_2), np.max(w_2))
        w_sorted = np.argsort(w_2)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model.get_hidden(data, path)
            # print(output[0].size())
            x = output[0][-1].cpu().numpy().T
            y = target.cpu().numpy()
            x_ += list(x[w_sorted[i]])
            y_ += list(y)
            # print(x.shape, y.shape)
            # plt.scatter(y,x[np.argmax(w_2)], c='b', alpha=0.1)
        #     corr = np.corrcoef([*x,y])[-1][:-1]
        #     plt.scatter(list(range(256)),corr,c='b',alpha=0.1)

        x_ = np.array(x_)
        y_ = np.array(y_)
        datasets = [x_[y_==t] for t in range(10)]
        datasets_sorted = sorted(datasets, key=lambda x: np.mean(x))
        x_ = sorted(x_)
        norm = [x_[6000*i:6000*(i+1)] for i in range(10)]
        # d = np.array(sorted(list(zip(x_, y_)), key=lambda t:t[0]))
        mat = []
        # for i in range(10):
        #     mat += Counter(d[i*6000:(i+1)*6000, 1])

        v = np.max(x_) - np.min(x_)
        # print(v)
        for p in range(10):
            k = norm[p]
            # k = k/np.std(k)
            mat+=[wasserstein_distance(datasets_sorted[p], k)]
        #print(mat, sum(mat)/v)
        print(sum(mat)/v)
        binned_data_sets = [
            np.histogram(d, range=hist_range, bins=100)[0]
            for d in datasets
            ]
        binned_maximums = np.max(binned_data_sets, axis=1)
        # print(binned_maximums)
        x_locations = np.arange(0, sum(binned_maximums), sum(binned_maximums)//10)
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
        # plt.show()
        plt.savefig('wasserstein_6/'+str(i)+'_'+str(sum(mat)/v)+'.png')
        plt.close()


def lesion_test(n, lesion):
    path = './model/new/' + str(n) + '_lstm.model'
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader
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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # model.get_hidden(data, path)
            # output = model.show_pred(data, path)
            # for i in range(target.size(0)):
            #     print(target[i].cpu().numpy(), output[1][i].cpu().numpy())
            output = model2(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        end.record()
        torch.cuda.synchronize()
        print("Time elapse: " + str(start.elapsed_time(end)))

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct/10000.


def get_wasser(n,i=0):
    path = './model/new/' + str(n) + '_lstm.model'
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader
    device = torch.device("cuda")
    x_ = []
    y_ = []

    with torch.no_grad():
        model = SequentialMNIST(64, n).to(device)
        model.load(path)
        pca = TruncatedSVD(n_components=2)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            hid = model.get_hidden(data)[0]
            lab = model.show_pred(data)[1]
            hid = hid.cpu().numpy() # 28*64*256
            lab = lab.cpu().numpy()
            print(lab.shape)
            hid = hid.reshape(28*64, 256)
            x2 = pca.fit_transform(hid)
            x2 = x2.reshape(28, 64, 2)
            for i in range(64):
                for j in range(28):
                    plt.scatter(x2[j,i,1], x2[j,i,0], c=colors[lab[i,j]])
                plt.plot(x2[:,i,1], x2[:,i,0], color='c',alpha=0.2)
            plt.show()


def get_embed(n, i=0):
    path = './model/new/' + str(n) + '_lstm.model'
    path_ = './model/auto/256.model'

    data = MNIST()

    train_loader = data.trainloader
    test_loader = data.testloader
    device = torch.device("cuda")
    x_ = []
    y_ = []

    with torch.no_grad():
        model = SequentialMNIST(64, n).to(device)
        model.load(path)
        auto = Autoencoder().to(device)
        auto = torch.load(path_)
        fig, ax = plt.subplots()
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        pca = TruncatedSVD(n_components=2)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            hid = model.get_hidden(data)[0]
            embed = auto.encoder(hid)
            # print(embed.size())
            lab = model.show_pred(data)[1]
            hid = hid.cpu().numpy()  # 28*64*256
            lab = lab.cpu().numpy()
            print(lab.shape)
            hid = hid.reshape(28 * 256, 256)
            # x2 = tsne.fit_transform(hid)
            # x2 = x2.reshape(28, 256, 2)
            # state = embed[-1,:]
            # state = x2[-1,:]
            # x2 = embed.cpu().numpy()
            dat = data.cpu().numpy()
            dat=dat.squeeze()
            dat = dat.reshape(256, 28*28)
            # print(dat.shape)
            d2 = tsne.fit_transform(dat)



            for i in range(256):
                # for j in range(28):
                #     plt.scatter(x2[j, i, 1], x2[j, i, 0], c=color[:, lab[i, j]])
                plt.scatter(d2[i,1], d2[i,0], c=colors[lab[i,-1]])
                # plt.scatter(state[i,1],state[i,0], c=colors[lab[i, -1]])
                # plt.plot(x2[:, i, 1], x2[:, i, 0], c=colors[lab[i, -1]], alpha=0.05)
            plt.legend(handles=legend_elements)
            plt.axis('off')
            plt.show()
            plt.clf()


def train_auto(model):
    data = np.loadtxt('states.csv')
    tensor = torch.stack([torch.Tensor(x) for x in data])
    dataset = utils.TensorDataset(tensor)
    dataloader = utils.DataLoader(dataset, batch_size=1000)
    device = torch.device("cuda")
    # model = Autoencoder()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.MSELoss()
    loss = None

    for d in dataloader:
        d = d[0].to(device)
        optimizer.zero_grad()
        output = model(d)
        # print(d.size(), output.size())
        loss = loss_function(output, d)
        loss.backward()
        optimizer.step()

    print('Loss is: ' + str(loss.item()))




def save_state():
    path = './model/new/256_lstm.model'
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader
    device = torch.device("cuda")
    d = np.zeros((28, 3200, 256))
    l = 0
    with torch.no_grad():
        model = SequentialMNIST(64, 256).to(device)
        model.load(path)
        pca = TruncatedSVD(n_components=2)
        for data, target in train_loader:

            data, target = data.to(device), target.to(device)

            hid = model.get_hidden(data)[0]
            # lab = model.show_pred(data)[1]
            hid = hid.cpu().numpy() # 28,64,256
            _,b,_ = hid.shape
            d[:,l:l+b,:] = hid
            l+=b
            if l>=3200:
                break

    d = d.reshape(28*3200, 256)
    np.savetxt('states.csv',d)

def get_hist(arr):
    a,b = arr.shape
    corr = np.zeros((b,b))
    for i in range(b):
        for j in range(i+1,b):
            d = wasserstein_distance(arr[:,i], arr[:,j])
            corr[i][j] = d
            corr[j][i] = d
    return corr

if __name__ == '__main__':
    # get_output(256)
    # for i in [4,6,8,10,12,16,32,64,128,256]:
    #     main(i)
    # lesion_test(256, 4)
    # a = []
    # b = []
    # for i in range(4,256):
    #     a += [lesion_test(256,i)]
    #     b += [4*i**2+130*i+10]
    # np.savetxt('comp_d.csv',[b,a])
    # plt.scatter(b,a)
    # plt.plot(b,a)
    # plt.xlim(np.max(b), np.min(b))
    #
    # plt.xscale('log', basex=10)
    # plt.xlabel('# of Params Remaining')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # # plt.legend()
    # plt.show()
    # for i in range(256):
    #     get_output(256,i)
    # lesion_test(256,16)
    # get_wasser(256)
    # save_state()
    # auto = Autoencoder().to(torch.device('cuda'))
    # for i in range(50):
    #     train_auto(auto)
    #
    # torch.save(auto, './model/auto/256.model')
    get_embed(256)