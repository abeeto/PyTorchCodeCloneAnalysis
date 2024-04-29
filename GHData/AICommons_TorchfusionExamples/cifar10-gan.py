from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator,StandardProjectionDiscriminator
from torch.optim import Adam
from torchfusion.datasets import cifar10_loader
import torch.cuda as cuda
import torch.nn as nn

G = StandardGenerator(output_size=(3,32,32),latent_size=128,num_classes=10)
D = StandardProjectionDiscriminator(input_size=(3,32,32),num_classes=10,apply_sigmoid=False)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())
    D = nn.DataParallel(D.cuda())

g_optim = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optim = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))

dataset = cifar10_loader(size=32,batch_size=64)

learner = RAvgHingeGanLearner(G,D)

if __name__ == "__main__":
    learner.train(dataset,gen_optimizer=g_optim,disc_optimizer=d_optim,num_classes=10,model_dir="./mnist-gan",latent_size=128,batch_log=False)