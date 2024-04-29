from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator,StandardProjectionDiscriminator
from torch.optim import Adam
from torchfusion.datasets import fashionmnist_loader
import torch.cuda as cuda
import torch.nn as nn
import torch

G = StandardGenerator(output_size=(1,32,32),latent_size=128)
D = StandardProjectionDiscriminator(input_size=(1,32,32),apply_sigmoid=False)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())
    D = nn.DataParallel(D.cuda())

g_optim = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optim = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))

dataset = fashionmnist_loader(size=32,batch_size=64)

class CustomGanLearner(StandardBaseGanLearner):
    def __update_discriminator_loss__(self, real_images, gen_images, real_preds, gen_preds):

        pred_loss = -torch.mean(real_preds - gen_preds)

        return pred_loss

    def __update_generator_loss__(self,real_images,gen_images,real_preds,gen_preds):

        pred_loss = -torch.mean(gen_preds - real_preds)
        return pred_loss

learner = CustomGanLearner(G,D)

if __name__ == "__main__":
    learner.train(dataset,gen_optimizer=g_optim,disc_optimizer=d_optim,model_dir="./mnist-gan-custom",latent_size=128,batch_log=True)