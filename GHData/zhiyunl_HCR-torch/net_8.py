"""
8-class classifier, 2 conv with pooling, 3 fully-connect
"""

from collections import Counter

import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.backends import cudnn

from MyLoader import *


# W - Input Width, F - Filter(or kernel) size, P - padding, S - Stride, Wout - Output width
# Wout = ((W−F+2P)/S)+1
class Net_8(nn.Module):
    def __init__(self):
        super(Net_8, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # input: 1 for grey, 3 for RBG, neuron nums, output=6, size= 52-4 =48
        self.pool = nn.MaxPool2d(2, 2)  # 48/2 =24
        self.conv2 = nn.Conv2d(6, 16, 5)  # 16*20*20 pool 16*10*10
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# for 6400/64 = 100 mini batches
def net8TrainOnce(net, trn_loader):
    running_loss = 0.0
    for i, (input, label) in enumerate(trn_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input)
        # print(outputs.is_cuda())
        label = label.to(device, dtype=torch.long)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_loss / 20))
            loss_print.append(running_loss / 20)
            running_loss = 0.0
    return net, loss_print


def getAcc(trueLabel, predict):
    # get accuracy
    error = np.array(predict) - np.array(trueLabel)
    # print(error.shape)
    c = Counter(error)
    return c[0] / error.shape[0]


def net8TestOnce(net, tst_loader):
    true_label = []
    predicted = []
    for i, (input, label) in enumerate(tst_loader):
        outputs = net(input)
        _, tmp = torch.max(outputs, 1)
        predicted.append(tmp)
        true_label.append(label)
        # print(input.shape)
    predicted = torch.stack(predicted, dim=0).to("cpu")
    true_label = torch.stack(true_label, dim=0)

    # print(predicted)
    getAcc(torch.flatten(true_label), torch.flatten(predicted))

    return torch.flatten(predicted).numpy()


def net8TestOn(net, dataset, device):
    predicted = []
    for i, input in enumerate(dataset):
        input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
        outputs = net(input)
        _, tmp = torch.max(outputs, 1)
        predicted.append(tmp)
        # print(input.shape, i)
    predicted = torch.stack(predicted, dim=0).to("cpu")
    return torch.flatten(predicted).numpy()


def plotTest():
    plt.figure()
    plt.title("CNN Accuracy")
    x = np.arange(testCnt)
    plt.plot(x, acc_print, 'b')
    plt.xlabel('test time')
    plt.ylabel('accuracy')
    plt.axis([0, 100, 0, 1])
    plt.show()


def plotTrain(epoch, loss_print):
    plt.figure()
    plt.title("CNN Loss")
    x = np.arange(epoch)
    plt.plot(x, loss_print, 'b')
    plt.xlabel('every 20 mini-batch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ('a', 'b', 'c', 'd', 'h', 'i', 'j', 'k')

    net = Net_8().to(device)  # init net
    batch_size = 1
    loss_print = []
    acc_print = []
    cnn_path = "./checkpoint"

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)  # .to(device)
    train = False
    testCnt = 1
    epoch = 100
    pkl_file = "train_data.pkl"
    npy_file = "finalLabelsTrain.npy"

    if train:
        data, label = loadFile(pkl_file, npy_file)
        data = preprocessor(data, 52, 52)
        for e in range(epoch):  # loop over the dataset multiple times

            trn_loader, tst_loader = split_trn_tst(data, label, batch_size=batch_size, trans=False, shuff=True,
                                                   whole=True)

            print(
                'Start Training epoch {}, batch size {}, remaining {} epochs'.format(e + 1, batch_size, epoch - e - 1))
            # create batch
            net = net8TrainOnce(net, trn_loader)
            # save model every epoch
            torch.save(net.state_dict(), "{}/epoch{}.pth".format(cnn_path, e))
            prediction = net8TestOnce(net, tst_loader)
        plotTrain()
        print('Finished Training')
    else:
        net.load_state_dict(torch.load("{}/epoch{}.pth".format(cnn_path, 99)))
        # net.load_state_dict(torch.load("net_8_final.pth"))
        # for cnt in range(testCnt):
        #     trn_loader, tst_loader = initLoader(batch_size, trans=False, shuff=False, whole=True,tst=True)
        #     net8TestOnce(net, tst_loader)
        # plotTest()
        data = load_pkl("HardData.pkl")
        data = np.load("wanyudata.npy", allow_pickle=True)
        label = load_npy("wanyulabels.npy")
        data = preprocessor(data, 52, 52)
        # predict_labels = net8TestOn(net, data[0:20])
        predict_labels = net8TestOn(net, data, device)
        getAcc(label, predict_labels)
