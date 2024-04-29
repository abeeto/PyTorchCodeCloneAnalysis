from __future__ import print_function
import torch

import torch.nn as nn
from torchvision import datasets, transforms
import lib.norm_layer

def norm_image(img):
    LS = len(img.shape)
    EPS = 1e-8
    img = img.float()
    if LS==3:
        C = img.shape[0]
        for c in range(C):
            imgc = img[c,:,:]
            mn = torch.mean(imgc)
            ss = torch.std(imgc, unbiased=True) + EPS
            img[c,:,:] = (imgc - mn)/ss
    else:
        mn = torch.mean(img)
        ss = torch.std(img, unbiased=True) + EPS
        img = (img - mn) / ss
    return img

def norm_data(data):
    if len(data.shape)>3:
        N = data.shape[0]
        for n in range(N):
            data[n,:] = norm_image((data[n,:]))
    else:
        data = norm_image(data)
    return data


class ResNetNunit(nn.Module):
    def __init__(self, in_chans=16,out_chans=32, depth=1):
        super(ResNetNunit,self).__init__()
        self.model = nn.ModuleList()
        self.cnv1 = nn.Conv2d(in_channels=in_chans,
                        out_channels=out_chans,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.model.append(self.cnv1)
        for d in range(depth):
            self.model.append(nn.Conv2d(in_channels=out_chans, out_channels=out_chans,kernel_size=3,padding=1))
            self.model.append(nn.BatchNorm2d(out_chans))
            self.model.append(nn.ReLU())
    def forward(self,x):
        cnv1 = self.model[0]
        #xin = self.cnv1(x)
        xin = cnv1(x)
        x = xin
        for lay in self.model[1:]:
            #x = lay.forward(x)
            x = lay(x)
        y = x + xin
        return y


class ResNet(nn.Module):
    def __init__(self, nchans=3, xsize=64,ysize=64, nfilters=16, nhids=100, ncls=1, res_depth=1):
        super(ResNet, self).__init__()
        self.xsize= xsize
        self.ysize = ysize
        self.nchans = nchans
        self.nfilters=nfilters
        self.nhids = nhids
        self.ncls = ncls
        self.res_depth = res_depth
        self.cnv_out_feats = 0
        self.net = nn.ModuleList()
        self.optimizer = None
        self.loss_fun = None
        self.cuda_flg = torch.cuda.is_available()
        self.norm_layer = True

    def create_net(self,  nfliters=None):
        if nfliters is None:
            nfliters = self.nfilters
        self.net  = nn.ModuleList()
        sz = min(self.xsize, self.ysize)

        if self.norm_layer:
            norm_layer = lib.norm_layer.NormLayer(self.cuda_flg)
            self.net.append(norm_layer)
        nf = self.nchans
        while sz >= 8:
            if len(self.net) > 0:
                res_unit = ResNetNunit(in_chans=nf,out_chans=nf*2, depth=self.res_depth)
            else:
                res_unit = ResNetNunit(in_chans=self.nchans,out_chans=nf*2, depth=self.res_depth)
            self.net.append(res_unit)
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.net.append(pool)
            nf *= 2
            sz //= 2
            self.cnv_out_feats = nf*sz*sz
        flatten = nn.Flatten()
        self.net.append(flatten)
        fc = nn.Linear(in_features=self.cnv_out_feats,out_features=self.nhids)
        self.net.append(fc)
        relu = nn.ReLU()
        self.net.append(relu)

        fc = nn.Linear(in_features=self.nhids, out_features=self.ncls)
        self.net.append(fc)
        if self.ncls==1:
            lsig = nn.LogSigmoid()
            self.net.append(lsig)

        else:
            lsf = nn.Softmax()
            self.net.append(lsf)
        lr = 1e-4
        if self.cuda_flg:
            self.cuda()
        params = self.parameters()
        self.optimizer = torch.optim.Adam(params=params, lr=lr)
        self.loss_fun = torch.nn.CrossEntropyLoss()

    def forward(self,x, should_norm=False):
        if self.cuda_flg:
            x = x.cuda()
        if not self.training and self.norm_layer is False:
            x = norm_data(x)
        for lay in self.net:
            x = lay(x)
        return x

    #def __call__(self,x):
    #    return self.forward(x)
    def fit(self, train_loader, epochs=10):
        if len(self.net) <= 0:
            self.create_net()
        self.train()

        LEN = len(train_loader)
        loss_fun = self.loss_fun
        batch_size = -1
        for iter in range(epochs):
            acc = 0
            tloss = 0
            sz = 0
            for batchid, (x, y) in enumerate(train_loader):
                #y_pred = self.__call__(x)
                y_pred = self(x)
                batch_size = x.shape[0]
                self.optimizer.zero_grad()
                #loss = self.loss_fun(y_pred,y)
                if self.cuda_flg:
                    y = y.cuda()
                loss = loss_fun(y_pred, y)
                amax = torch.argmax(y_pred, dim=1)
                ids = amax == y
                ids = ids.cpu()
                """
                for i in range(batch_size):
                    if ids[i]:
                        acc += 1
                """
                acc += ids.sum()
                sz += batch_size
                tloss += loss
                loss.backward()
                self.optimizer.step()
                accf = float(acc) / sz
                print(f'\r batchid:{batchid}, loss:{loss}, acc:{accf}', end='')

            tloss /= LEN
            accf = float(acc) / sz
            print(f'\n----> epoch:{iter}/{epochs}, loss:{tloss}, acc:{accf}')
        device = 'cuda' if self.cuda_flg else 'cpu'
        print(f'End of train on device: {device}, accuracy: {accf}')

    def compute_acc(self, data, test_name):
        self.cuda()
        acc = 0
        sz = 0
        batch_size = -1
        for batchid, (x, y) in enumerate(data):
            #y_pred = self.__call__(x)
            y_pred = self(x)
            batch_size = x.shape[0]

            y = y.cuda()

            amax = torch.argmax(y_pred, dim=1)
            ids = amax == y
            ids = ids.cpu()
            acc += ids.sum()
            """
            for i in range(batch_size):
                if ids[i]:
                    acc += 1
            """

            sz += batch_size
 
        accf = float(acc) / sz
        print(f'{test_name}: accuracy: {accf}')
        return accf
    def save_model(self, file):
        torch.save(self, file)

    #classmethod
    def load_model(file):
        rnet = torch.load(file)
        return rnet


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description='MNIst example')
    ap.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch')
    ap.add_argument('--epochs', dest='epochs', type=int, required=True)
    ap.add_argument('--res_depth', dest='res_depth', type=int, default=3)
    arg = ap.parse_args()
    return arg

# git token: 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

if __name__ == '__main__':
    args = parse_args()
    import CustomDataLoader
    cdl = CustomDataLoader.CustomDataLoader(norm_flag=False)
    dl = cdl.get_loader()
    rnet = ResNet(nchans=1, xsize=28,ysize=28,ncls=10, res_depth=args.res_depth)
    cuda_flg = torch.cuda.is_available()
    print(f'cuda_flg={cuda_flg}')
    rnet.fit(dl, epochs=args.epochs)

    models_path = 'C:/Users/shimon.cohen/data/torch/resnet.model'
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                                                              transform=transforms.Compose(
                                                                  [transforms.ToTensor()]
                                                              )),
                                               batch_size=args.batch_size, shuffle=False)
    rnet.eval()
    acc = rnet.compute_acc(test_loader, 'test')
    #torch.save(rnet, models_path)
    #rnet = torch.load(models_path)
    rnet.save_model(models_path)
    rnet1 = ResNet.load_model(models_path)
    rnet1.eval()
    print('test set')
    acc = rnet1.compute_acc(test_loader, 'test')


