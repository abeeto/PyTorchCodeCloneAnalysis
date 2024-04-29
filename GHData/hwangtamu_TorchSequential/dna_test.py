import csv, random
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from src.LSTM import SequentialMNIST
import torch
import torch.optim as optim
import torch.nn.functional as F


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

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(data.size())
        # print(torch.reshape(target,(64, -1)))
        # y_onehot = torch.cuda.FloatTensor(output.size(0),output.size(1))
        # y_onehot.zero_()
        # y_onehot.scatter_(1, torch.reshape(target,(target.size(0), -1)), 1)
        # print(output, y_onehot)
        # loss = loss_function(output, y_onehot)

        #loss = loss_function(output, Variable(target))
        loss = F.nll_loss(output, torch.max(target, 1)[1], reduction='sum')
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            target = torch.max(target, 1)[1]
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)

def main(hidden=256):
    epochs = 1000

    splice = Splice('./DNA/splice.data')

    train_ = TensorDataset(splice.train['x'], splice.train['y'])
    test_ = TensorDataset(splice.test['x'], splice.test['y'])
    train_loader = DataLoader(train_, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_,batch_size=32, shuffle=False)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    path = './model/dna/' + str(hidden) + '_gru.model'
    model = SequentialMNIST(batch_size=64,
                            hidden_size=hidden,
                            in_size=4,
                            out_size=3).to(device)
    model.load(path)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # for epoch in range(1, epochs + 1):
    #     train(model, device, train_loader, optimizer, epoch)
    #     if epoch%10==0:
    #         test(model, device, test_loader)
    #
    # torch.save(model, './model/dna/'+str(hidden)+'_gru_d.model')
    return test(model, device, test_loader)

a = []
for i in [4,6,8,10,12,16,32,64,128,256]:
    a+=[main(i)]
print(sorted(a))