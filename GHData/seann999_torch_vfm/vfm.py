import torch
import torch.nn as nn
from torch.autograd import Variable

s=32*11*11
size = 11
rnn_layers = 1

var = True

class VFM(nn.Module):

    def __init__(self, img_size=64, action_size=2, z_size=32, rnn_input_size=256, rnn_size=512, batch_size=32,
                 in_len=10, out_len=10):
        super(VFM, self).__init__()

        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.img_size = img_size
        self.in_len = in_len
        self.out_len = out_len
        self.action_size = action_size
        self.reconstruction_function = nn.BCELoss()
        self.reconstruction_function.size_average = False
        self.z_size = z_size

        self.en = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=(0,0)),
            torch.nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=6, stride=2, padding=(1,1)),
            torch.nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=6, stride=2, padding=(1,1)),
            torch.nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=(0,0)),
            torch.nn.ELU(),
            nn.BatchNorm2d(32),
        )

        self.en2 = torch.nn.Sequential(
            nn.Linear(s, rnn_input_size),
            torch.nn.ELU(),
            nn.BatchNorm1d(rnn_input_size),
        )

        self.h2h = nn.Linear(rnn_size, rnn_size)

        self.a2h = nn.Linear(action_size, rnn_size)

        self.rnn_cell = nn.LSTMCell(rnn_input_size, rnn_size)
        #self.rnn = nn.LSTM(rnn_input_size, rnn_size, rnn_layers, batch_first=True)
        #self.rnn2 = nn.LSTM(rnn_input_size, rnn_size, rnn_layers, batch_first=True)

        if var:
            self.mean = nn.Linear(rnn_size, z_size)
            self.logvar = nn.Linear(rnn_size, z_size)

        if var:
            self.de = torch.nn.Sequential(
                nn.BatchNorm1d(z_size),
                nn.Linear(z_size, s),
                torch.nn.ELU(),
                nn.BatchNorm1d(s),
            )
        else:
            self.de = torch.nn.Sequential(
                nn.BatchNorm1d(z_size),
                nn.Linear(rnn_size, s),
                torch.nn.ELU(),
                nn.BatchNorm1d(s),
            )

        self.de2 = torch.nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=(0,0)),
            torch.nn.ELU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=6, stride=2, padding=(1,1)),
            torch.nn.ELU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 64, kernel_size=6, stride=2, padding=(1,1)),
            torch.nn.ELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 6, kernel_size=8, stride=2, padding=(0,0)),
            torch.nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).cuda()
        return eps.mul(std).add_(mu)

        #        | | | | |
        #+-+-+-+-+-+-+-+-+
        #| | | | | 

    def encode(self, x, acts_encode, acts_decode):
        #print(x.size())
        x = x.view(self.batch_size*self.in_len, 3, self.img_size, self.img_size)
        h = self.en(x)

        img_enc = h.view(-1, s)
        img_enc = self.en2(img_enc)

        rnn_h = Variable(torch.zeros(self.batch_size, self.rnn_size)).cuda()
        rnn_c = Variable(torch.zeros(self.batch_size, self.rnn_size)).cuda()
        img_enc = img_enc.view(self.batch_size, self.in_len, -1)

        for i in range(self.in_len):
            rnn_h, rnn_c = self.rnn_cell(img_enc[:, i, :], (rnn_h, rnn_c))
            rnn_h = self.h2h(rnn_h) * self.a2h(acts_encode[:, i, :])

        gen_imgs = []
        means = []
        logvars = []
        rnn_h = Variable(rnn_h.data, requires_grad=False)

        if var:
            mean = self.mean(rnn_h)
            logvar = self.logvar(rnn_h)
            means.append(mean)
            logvars.append(logvar)
            img_pred = self.de(self.reparameterize(mean, logvar))
        else:
            img_pred = self.de(h)

        img_pred = self.de2(img_pred.view(self.batch_size, 32, size, size))
        gen_imgs.append(img_pred)
        img_pred = Variable(img_pred[:, :3, ...].data, requires_grad=False)
        #print(img_pred.size())
        img_enc = self.en(img_pred)
        img_enc = self.en2(img_enc.view(-1, s))

        for i in range(self.out_len-1):
            rnn_h = self.h2h(rnn_h) * self.a2h(acts_decode[:, i, :])
            rnn_h, rnn_c = self.rnn_cell(img_enc, (rnn_h, rnn_c))

            if var:
                mean = self.mean(rnn_h)
                logvar = self.logvar(rnn_h)
                means.append(mean)
                logvars.append(logvar)
                img_pred = self.de(self.reparameterize(mean, logvar))
            else:
                img_pred = self.de(rnn_h)

            img_pred = self.de2(img_pred.view(self.batch_size, 32, size, size))
            gen_img = img_pred
            gen_imgs.append(gen_img)
            img_pred = Variable(gen_img[:, :3, ...].data, requires_grad=False)
            img_enc = self.en(img_pred)
            img_enc = self.en2(img_enc.view(-1, s))

        gen_imgs = torch.stack(gen_imgs, 1)
        mean = torch.stack(means, 0)
        logvar = torch.stack(logvars, 0)
        #mean = mean.view(self.out_len, self.batch_size, -1)
        #logvar = logvar.view(self.out_len, self.batch_size, -1)

        #print(mean)

        return gen_imgs, mean, logvar

    def forward(self, x, acts_encode, acts_decode):
        out, mean, logvar = self.encode(x, acts_encode, acts_decode)

        return out, mean, logvar

    def loss(self, x, recon, mean, logvar):
        a = recon[:, :, :3, ...].contiguous().view(-1, 3, 210, 210)
        b = x.view(-1, 3, 210, 210)
        mse = self.reconstruction_function(a, b)
        #mse = (b.mul(a.log()).add(b.mul(-1).add(1).mul(a.log().mul(-1).add(1))))
        
        #mse = mse.sum(1).mean()
        #print(mse.data)
        #mse = (x - recon[:, :, :3, ...])**2.0
        #mse = (a + recon[:, :, 3:, ...]).mul(0.5)
        #mse = mse.sum(1).mean()#.sum(2).sum(3).sum(4).mean()

        mean = mean.view(-1, self.z_size)
        logvar = logvar.view(-1, self.z_size)
        KLD_element = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = KLD_element.sum().mul_(-0.5)
        return mse + KLD, mse, KLD
