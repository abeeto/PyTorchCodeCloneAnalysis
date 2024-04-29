# import all the required torch and torchvision modules.
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# import all the other non-torch modules.
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#################### SET THE RANDOM SEED #############################################

torch.manual_seed(0)

##################### IMPORT THE DATASET ##############################################

TRAINING_DIR = "../../datasets/mnist-jpg/trainingSet/trainingSet"
TEST_DIR = "../../datasets/mnist-jpg/testSet"
trans = transforms.Compose([
  transforms.Grayscale(),
  transforms.Resize(28),
  transforms.ToTensor()
])

trainfolder = datasets.ImageFolder(root=TRAINING_DIR, transform=trans)
testfolder = datasets.ImageFolder(root=TEST_DIR, transform=trans)

trainloader = data.DataLoader(trainfolder, batch_size=128, shuffle=True, num_workers=6)
testloader = data.DataLoader(testfolder, batch_size=128, shuffle=True, num_workers=6)

############################ CREATE THE VAE MODEL #####################################

class MNISTVAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Linear(784, 400),
      nn.ReLU(True)
    )
    self.mu_layer = nn.Linear(400, 20)
    self.logvar_layer = nn.Linear(400, 20)
    self.decoder = nn.Sequential(
      nn.Linear(20, 400),
      nn.ReLU(True),
      nn.Linear(400, 784),
      nn.Sigmoid()
    )

  def __reparam__(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.rand_like(std)
    return mu + std * epsilon

  def encode(self, x):
    x = self.encoder(x)
    mu, logvar = self.mu_layer(x), self.logvar_layer(x)
    return x, mu, logvar

  def decode(self, x):
    x = self.decoder(x)
    return x

  def forward(self, x):
    x, mu, logvar = self.encode(x)
    z = self.__reparam__(mu, logvar)
    x = self.decoder(z)
    return x, mu, logvar

################# CREATE THE LOSS FUNCTION ##############################################

def loss_function(x_pred, x, mu, logvar, epoch):
  kl_weight = epoch / 400.

  mse = fn.mse_loss(x_pred, x)
  kl = kl_weight * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
  return mse - kl

################ INIT THE MODEL AND THE OPTIMIZER #######################################

vae = MNISTVAE().cuda()
adam = optim.Adam(vae.parameters(), lr=1e-3)

################ TRAIN THE MODEL ########################################################

for epoch in range(200):
  for x, _ in tqdm(trainloader):
    adam.zero_grad()

    x = x.cuda().float().view(-1, 784)

    x_pred, mu, logvar = vae.forward(x)

    train_loss = loss_function(x_pred, x, mu, logvar, epoch)

    train_loss.backward()
    adam.step()

  print("\n")
  print("[{}] Train Loss={}".format(epoch+1, train_loss.detach().cpu().numpy()))
  print("\n")

  sample = torch.randn(128, 20).cuda().float()
  generated_sample = vae.decode(sample).detach().cpu().numpy()[0]
  generated_sample = np.moveaxis(generated_sample, 0, -1)
  plt.imshow(generated_sample.reshape(28, 28), cmap='gray')
  plt.savefig("images/generated_image_epoch_{}.png".format(epoch+1))
      
