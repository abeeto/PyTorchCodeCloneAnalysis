import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt

from ConvMNISTVAE import *
from LossFunction import *

ROOT_DIR = "../../datasets/mnist-jpg"
TRAINING_DIR = os.path.join(ROOT_DIR, "trainingSet", "trainingSet")

trans = transforms.Compose([
  transforms.Grayscale(),
  transforms.Resize(28),
  transforms.ToTensor()
])

trainfolder = datasets.ImageFolder(root=TRAINING_DIR, transform=trans)
trainloader = data.DataLoader(trainfolder, batch_size=128, shuffle=True, num_workers=6)

vae = ConvMNISTVAE().cuda()
adam = optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(50):
  for x, _ in tqdm(trainloader):
    x = x.cuda().float()

    adam.zero_grad()

    x_preds, mu, logvar = vae.forward(x)
    loss = loss_function(x_preds.view(-1, 784), x.view(-1, 784), mu, logvar)

    loss.backward()
    adam.step()

  print("\n")
  print("[{}] Loss={}".format(epoch+1, loss.detach().cpu().numpy()))
  print("\n")

  sample = torch.randn(128, 20).cuda().float()
  generated_images = vae.decode(sample)
  generated_image = np.moveaxis(generated_images.detach().cpu().numpy()[0], 0, -1)
  plt.imshow(generated_image.reshape(28, 28), cmap='gray')
  plt.savefig("images/generated_epoch_{}.png".format(epoch+1))
