import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from BasicMNISTVAE import *
from LossFunction import *

ROOT_DIR = "../../datasets/mnist-jpg"
TRAINING_DIR = os.path.join(ROOT_DIR, "trainingSet", "trainingSet")
TEST_DIR = os.path.join(ROOT_DIR, "testSet")

trans = transforms.Compose([
  transforms.Grayscale(),
  transforms.Resize(28),
  transforms.ToTensor()
])

trainfolder = datasets.ImageFolder(root=TRAINING_DIR, transform=trans)
testfolder = datasets.ImageFolder(root=TEST_DIR, transform=trans)
trainloader = data.DataLoader(trainfolder, batch_size=128, shuffle=True, num_workers=6)
testloader = data.DataLoader(testfolder, batch_size=1, shuffle=True, num_workers=6)

vae = BasicMNISTVAE().cuda()
sgd = optim.Adam(vae.parameters(), lr=1e-3)

# TODO : Train the model!
# TODO : Push the script to a Github repo and a paperspace gradient instance

for epoch in range(50):
  for x, _ in tqdm(trainloader):
    sgd.zero_grad()

    x = x.cuda().float().view(-1, 784)
    x_pred, mu, logvar = vae.forward(x)

    loss = loss_function(x_pred, x, mu, logvar)
    
    loss.backward()
    sgd.step()

  print("[{}] Loss={}".format(epoch+1, loss.detach().cpu().numpy()))
  

  random_sample = torch.randn(128, 60).cuda().float()
  random_image = vae.decode(random_sample)
  plt.imshow(np.moveaxis(random_image.detach().cpu().numpy()[0], 0, -1).reshape(28, 28), cmap='gray')
  plt.savefig("images/generated_epoch_{}.png".format(epoch+1)) 
  if epoch + 1 % 5 == 0:
    # save the model.
    torch.save(vae.state_dict(), "models/epoch_{}_weights.pth".format(epoch+1))
