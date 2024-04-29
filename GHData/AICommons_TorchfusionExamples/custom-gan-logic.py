from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator,StandardProjectionDiscriminator
from torch.optim import Adam
from torchfusion.datasets import fashionmnist_loader
import torch.cuda as cuda
import torch.nn as nn
import torch
from torch.autograd import Variable

G = StandardGenerator(output_size=(1,32,32),latent_size=128)
D = StandardProjectionDiscriminator(input_size=(1,32,32))

if cuda.is_available():
    G = nn.DataParallel(G.cuda())
    D = nn.DataParallel(D.cuda())

g_optim = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optim = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))

dataset = fashionmnist_loader(size=32,batch_size=64)


class CustomGanLearner(BaseGanCore):

    def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,loss_fn=nn.BCELoss(),**kwargs):

        self.latent_size = latent_size
        self.loss_fn = loss_fn
        super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,**kwargs)

    def __disc_train_func__(self, data):

        super().__disc_train_func__(data)

        self.disc_optimizer.zero_grad()

        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        batch_size = x.size(0)

        source = self.dist.sample((batch_size,self.latent_size))

        real_labels = torch.ones(batch_size,1)
        fake_labels = torch.zeros(batch_size,1)

        if self.cuda:
            x = x.cuda()
            source = source.cuda()
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()

        x = Variable(x)
        source = Variable(source)

        outputs = self.disc_model(x)

        generated = self.gen_model(source)
        gen_outputs = self.disc_model(generated.detach())

        gen_loss = self.loss_fn(gen_outputs,fake_labels)

        real_loss = self.loss_fn(outputs,real_labels)

        loss = gen_loss + real_loss
        loss.backward()
        self.disc_optimizer.step()

        self.disc_running_loss.add_(loss.cpu() * batch_size)

    def __gen_train_func__(self, data):

        super().__gen_train_func__(data)

        self.gen_optimizer.zero_grad()

        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        batch_size = x.size(0)

        source = self.dist.sample((batch_size,self.latent_size))

        real_labels = torch.ones(batch_size,1)

        if self.cuda:
            source = source.cuda()
            real_labels = real_labels.cuda()

        source = Variable(source)

        fake_images = self.gen_model(source)
        outputs = self.disc_model(fake_images)

        loss = self.loss_fn(outputs,real_labels)
        loss.backward()

        self.gen_optimizer.step()

        self.gen_running_loss.add_(loss.cpu() * batch_size)


learner = CustomGanLearner(G,D)

if __name__ == "__main__":
    learner.train(dataset,gen_optimizer=g_optim,disc_optimizer=d_optim,model_dir="./mnist-gan-custom-logic",latent_size=128,batch_log=True)