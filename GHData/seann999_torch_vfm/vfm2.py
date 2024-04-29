import torch
import torch.nn as nn
from torch.autograd import Variable

s=32*11*11
size = 11
rnn_layers = 1

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
        #self.reconstruction_function = nn.BCELoss()
        #self.reconstruction_function.size_average = False
        self.z_size = z_size

        self.q_en = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=(0,0)),
            torch.nn.ELU(),
            #nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=6, stride=2, padding=(1,1)),
            torch.nn.ELU(),
            #nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=6, stride=2, padding=(1,1)),
            torch.nn.ELU(),
            #nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=(0,0)),
            torch.nn.ELU(),
            #nn.BatchNorm2d(32)
        )

        self.q_en2 = torch.nn.Sequential(
            nn.Linear(s, rnn_input_size),
            torch.nn.ELU(),
            nn.BatchNorm1d(rnn_input_size),
        )

        self.rnn_cell = nn.GRUCell(rnn_input_size + action_size, rnn_size)

        self.qz_mean = torch.nn.Sequential(
            nn.Linear(rnn_size, 256),
            torch.nn.ELU(),
            nn.Linear(256, z_size)
        )
        self.qz_logvar = torch.nn.Sequential(
            nn.Linear(rnn_size, 256),
            torch.nn.ELU(),
            nn.Linear(256, z_size)
        )

        self.pz_en = torch.nn.Sequential(
            nn.Linear(z_size + action_size, 256),
            torch.nn.ELU(),
            nn.Linear(256, 256),
            torch.nn.ELU(),
            #nn.BatchNorm1d(256),
        )

        self.pz_mean = nn.Linear(256, z_size)
        self.pz_logvar = nn.Linear(256, z_size)

        self.de = torch.nn.Sequential(
            nn.Linear(z_size, s),
            torch.nn.ELU(),
            #nn.BatchNorm1d(s),
        )

        self.de2 = torch.nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=(0,0)),
            torch.nn.ELU(),
            #nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=6, stride=2, padding=(1,1)),
            torch.nn.ELU(),
            #nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 64, kernel_size=6, stride=2, padding=(1,1)),
            torch.nn.ELU(),
            #nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=8, stride=2, padding=(0,0)),
            torch.nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).cuda()
        return eps.mul(std).add_(mu)

    def encode(self, x, acts_encode, acts_decode):
        h = self.q_en(x.view(self.batch_size*self.in_len, 3, self.img_size, self.img_size))

        img_enc = h.view(-1, s)
        img_enc = self.q_en2(img_enc)
 
        q_means = []
        q_logvars = []
        gen_imgs = []
        pred_imgs = []

        rnn_h = Variable(torch.zeros(self.batch_size, self.rnn_size)).cuda()
        #rnn_c = Variable(torch.zeros(self.batch_size, self.rnn_size)).cuda()
        img_enc = img_enc.view(self.batch_size, self.in_len, -1)

        for i in range(self.in_len):
            rnn_h = self.rnn_cell(torch.cat([img_enc[:, i, :], acts_encode[:, i, :]], 1), rnn_h)
            mean = self.qz_mean(rnn_h)
            logvar = self.qz_logvar(rnn_h)
            z = self.reparameterize(mean, logvar)
            #q_zs.append(z)
            q_means.append(mean)
            q_logvars.append(logvar)

            img_pred = self.de(z)
            img_pred = self.de2(img_pred.view(self.batch_size, 32, size, size))
            #gen_imgs.append(img_pred)
            gen_imgs.append(img_pred)

        q_z = z

        p_means = []
        p_logvars = []
        p_zs = []
        curr_z = self.reparameterize(Variable(torch.zeros(self.batch_size, self.z_size).float()).cuda(),
            Variable(torch.zeros(self.batch_size, self.z_size)).cuda())

        for i in range(self.in_len):
            h = self.pz_en(torch.cat([curr_z, acts_encode[:, i, :]], 1))
            mean = self.pz_mean(h)
            logvar = self.pz_logvar(h)
            curr_z = self.reparameterize(mean, logvar)
            p_zs.append(curr_z)
            p_means.append(mean)
            p_logvars.append(logvar)

        for i in range(self.out_len):
            h = self.pz_en(torch.cat([q_z, acts_decode[:, i, :]], 1))
            mean = self.pz_mean(h)
            logvar = self.pz_logvar(h)
            curr_z = self.reparameterize(mean, logvar)

            img_pred = self.de(curr_z)
            img_pred = self.de2(img_pred.view(self.batch_size, 32, size, size))
            pred_imgs.append(img_pred)

        gen_imgs = torch.stack(gen_imgs, dim=1)
        pred_imgs = torch.stack(pred_imgs, dim=1)
        q_means = torch.stack(q_means, dim=1)
        q_logvars = torch.stack(q_logvars, dim=1)
        p_means = torch.stack(p_means, dim=1)
        p_logvars = torch.stack(p_logvars, dim=1)

        return gen_imgs, pred_imgs, q_means, q_logvars, p_means, p_logvars

    def forward(self, x, acts_encode, acts_decode):
        return self.encode(x, acts_encode, acts_decode)

    def kld(self, m1, v1, m2, v2):
        s1 = v1.exp()
        s2 = v2.exp()

        return torch.log(s2) - torch.log(s1) + (s1.pow(2)+(m1-m2).pow(2))/(2.0*s2.pow(2)+1e-5) - 0.5

    def loss(self, x, recon, q_means, q_logvars, p_means, p_logvars):
        bce = -(x * torch.log(recon) + (1.0 - x) * torch.log(1.0 - recon))

        def reform(xx):
            return Variable(xx.data).cuda()#Variable(xx.view(-1, self.z_size).data).cuda()

        q_means = reform(q_means)
        q_logvars = reform(q_logvars)
        #p_means = p_means.view(-1, self.z_size)
        #p_logvars = p_logvars.view(-1, self.z_size)

        kld = self.kld(q_means, q_logvars, p_means, p_logvars)

        bce = bce.sum(4).sum(3).sum(2).mean(1).mean(0)
        kld = kld.sum(2).sum(1).mean(0)
        #KLD = KLD * 1e2
        #bce = bce * 1e-2

        #print(KLD)

        return bce + kld, bce, kld
