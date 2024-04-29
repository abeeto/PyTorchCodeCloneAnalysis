import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms

class ResBlock(LightningModule):
    def __init__(self, downsample, in_chan, out_chan, depth=3,  **kwargs):
        super(ResBlock,self).__init__(**kwargs)
        self.downsample = downsample
        self.models = torch.nn.ModuleList()
        self.create(in_chan, out_chan,depth=depth)

    def create(self, in_chans, out_chans, depth=3):
        if self.downsample is not None or in_chans!=out_chans:
            stride = 1 if self.downsample is None else 2
            cnv = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, padding=1)
            self.downsample = cnv
            self.models.append(cnv)
        for i in range(depth-1):
            cnv = nn.Conv2d(in_channels=out_chans, out_channels=out_chans, kernel_size=3, padding=1)
            self.models.append(cnv)
            bn = nn.BatchNorm2d(out_chans)
            self.models.append(bn)
            relu = nn.ReLU()
            self.models.append(relu)
        # add last one
        cnv = nn.Conv2d(in_channels=out_chans, out_channels=out_chans, kernel_size=3, padding=1)
        self.models.append(cnv)
        bn = nn.BatchNorm2d(out_chans)
        self.models.append(bn)

    def forward(self, X):
        xin = self.models[0](X)
        residual = X
        if self.downsample:
            residual = xin
        L = len(self.models)
        x = xin
        for k in range(1,L):
            m = self.models[k]
            x = m(x)
        y = x + residual
        y = F.relu(y)
        return y

class LResNet(LightningModule):
    def __init__(self, sizex, sizey, nchan, min_size=8, nfliters=32,nhids=100, ncls=-1, res_depth=3,  **kwargs):
        super(LResNet,self).__init__(**kwargs)
        self.sizex = sizex
        self.sizey = sizey
        self.esx,self.esy=None, None
        self.nchan = nchan
        self.nfilters = nfliters
        self.models = torch.nn.ModuleList()
        self.min_size = min_size
        self.ncls = ncls
        self.lin1 = None
        self.lin2 = None
        self.nhids = nhids
        self.res_depth = res_depth
        if ncls>0:
            self.create(sizex, sizey,nchan,ncls)
    def create(self, sizex, sizey,nchan,ncls):
        sx,sy = self.sizex, self.sizey
        nf = self.nfilters
        nchan = nchan
        while sx > self.min_size and sy > self.min_size:
            rblock = ResBlock(True, in_chan=nchan, out_chan= nf, depth=self.res_depth)
            nchan = nf
            nf *= 2
            self.models.append(rblock)
            sx = int(sx//2)
            sy = int(sy//2)
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.models.append(pool)

        self.sex, self.esy = sx, sy
        nf //= 2
        lin_in = sx*sy*nf
        if self.nhids>0:
            self.lin1 = nn.Linear(in_features=lin_in, out_features=self.nhids)
            lin_in = self.nhids
        self.lin2 = nn.Linear(in_features=lin_in, out_features=self.ncls)

        params = self.parameters()
        self.optimizer = None
        self.loss_fun = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        for m in self.models:
            X = m(X)

        X = X.view(X.shape[0],-1)
        if self.lin1:
            X = self.lin1(X)
            X = F.relu(X)
        X = self.lin2(X)
        Y = F.softmax(X, dim=1)
        return Y

    def training_step(self, batch, bidx):
        X,Y = batch
        Y_p = self.forward(X)
        loss = self.loss_fun(Y_p, Y)
        acc = self.compute_accuracy(X,Y)
        if (bidx+1)%100==0:
            self.log('Accuracy:', acc, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log('CEloss:', loss,  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    """
    def validation_step(self, batch, bidx):
        X,Y = batch
        acc = self.compute_accuracy(X,Y)
        self.log('Validation_accuracy:',acc)
        return acc
    """
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return self.optimizer

    def compute_accuracy(self,x,y):
        yp = self(x)
        yp_max = torch.argmax(yp, dim=1)
        acck = torch.sum(yp_max == y)
        n = x.shape[0]
        acc = float(acck) / n
        return acc

    def evaluate(self, data):
        self.eval()
        acc = 0
        n = 0
        for x,y in data:
            yp = self(x)
            yp_max = torch.argmax(yp, dim=1)
            acck = torch.sum(yp_max == y)
            n += x.shape[0]
            acc += acck
        acc = float(acc)/float(n)
        return acc



    def train_on_data(self, train_data, valid_data, epochs=10):
        gpu_flag = torch.cuda.is_available()
        trainer = Trainer(gpus=1, max_epochs=epochs) if gpu_flag else Trainer(max_epochs=epochs)
        logger = trainer.logger
        trainer.fit(self, train_data, valid_data)



if __name__ == '__main__':
    lres_net = LResNet(28,28,1, nfliters=8, ncls=10,res_depth=7)
    # transforms
    # prepare transforms standard to MNIST
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # data
    mnist_train = MNIST('../data', train=True, download=True, transform=transform)
    mnist_loader = DataLoader(mnist_train, batch_size=64)
    mnist_train = MNIST('../data', train=False, download=True, transform=transform)
    mnist_valid = DataLoader(mnist_train, batch_size=64)
    lres_net.train_on_data(mnist_loader, mnist_valid, epochs=25)
    acc_train = lres_net.evaluate(mnist_loader)
    acc_valid = lres_net.evaluate(mnist_valid)
    print(f"train-accuracy={acc_train}\ttest_accuracy={acc_valid}")
    print(f"net:{lres_net}")

