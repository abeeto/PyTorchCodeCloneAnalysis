
import sys
import torch
from torch.nn import functional as F




class __loss():

    def __init__(self):
        pass

    def D(self, train, real_sample):
        if train.optimizer['optimizer type']=='lsgan':
            return self.lsgan_D(train, real_sample)
        if train.optimizer['optimizer type']=='logistic_loss':
            return self.logistic_D(train, real_sample)
        if train.optimizer['optimizer type']=='logistic_loss_ns':
            return self.logistic_D_ns(train, real_sample)
        if train.optimizer['optimizer type']=='logistic_loss_r1':
            return self.logistic_D_r1(train,real_sample)

    def G(self, train):
        if train.optimizer['optimizer type']=='lsgan':
            return self.lsgan_G(train)
        if train.optimizer['optimizer type']=='logistic_loss' or train.optimizer['optimizer type']=='logistic_loss_ns' or train.optimizer['optimizer type']=='logistic_loss_r1':
            return self.logistic_G(train)

    def lsgan_D(self, train, real_sample):

        batch_size = real_sample.size(0)
        real_label = 1.0
        fake_label = 0.0

        labels_r = torch.full((batch_size,), real_label, dtype=torch.float, device=train.device)
        output_r = train.out_D(real_sample).view(-1)
        loss_D_real = train.loss['loss criterion of D'](output_r, labels_r)

        labels_f= torch.full((batch_size,), fake_label, dtype=torch.float, device=train.device)
        output_f = train.out_D(train.out_G(batch_size).detach()).view(-1)
        loss_D_fake = train.loss['loss criterion of D'](output_f, labels_f)
        
        return (loss_D_real + loss_D_fake)*0.5

    def lsgan_G(self, train):

        real_label = 1.0

        labels = torch.full((train.batch_size,), real_label, dtype=torch.float, device=train.device)
        fakes = train.out_G(train.batch_size)
        output = train.out_D(fakes).view(-1)
        loss_G = train.loss['loss criterion of G'](output, labels)

        return loss_G

    def logistic_D(self, train, real_sample):

        batch_size = real_sample.size(0)
        fakes = train.out_G(batch_size).detach()
        out_real = train.out_D(real_sample).view(-1)
        out_fake = train.out_D(fakes).view(-1)
        loss_real = F.softplus(-out_real)
        loss_fake = F.softplus(out_fake)
        
        return loss_real.mean() + loss_fake.mean()

    def logistic_D_ns(self, train, real_sample):

        batch_size = real_sample.size(0)
        fakes = train.out_G(batch_size).detach()
        out_real = train.out_D(real_sample).view(-1)
        out_fake = train.out_D(fakes).view(-1)
        loss_real = -F.softplus(out_real)
        loss_fake = F.softplus(out_fake)
        
        return loss_real.mean() + loss_fake.mean()

    def logistic_D_r1(self, train, real_sample,gamma=10.0):

        real_sample = real_sample
        batch_size = real_sample.size(0)
        fakes = train.out_G(batch_size).detach()
        out_real = train.out_D(real_sample).view(-1)
        out_fake = train.out_D(fakes).view(-1)
        loss_real = F.softplus(-out_real)
        loss_fake = F.softplus(out_fake)

        gp_input = torch.autograd.Variable(real_sample, requires_grad=True)
        gp_output = train.out_D(gp_input)
        grad_outputs = torch.ones(gp_output.size())
        real_grads, = torch.autograd.grad(outputs=gp_output,inputs=gp_input,grad_outputs=grad_outputs,create_graph=True)
        gradient_penalty = real_grads.pow(2).sum()
        reg = gradient_penalty * (gamma * 0.5)
        reg.backward()
        
        return loss_real.mean() + loss_fake.mean()

    def logistic_G(self, train):

        fakes = train.out_G(train.batch_size)
        output = train.out_D(fakes).view(-1)
        loss_G = F.softplus(-output)

        return loss_G.mean()



sys.modules[__name__] = __loss()