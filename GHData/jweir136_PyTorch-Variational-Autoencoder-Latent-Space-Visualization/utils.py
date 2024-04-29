import matplotlib.pyplot as plt
import numpy as np
import torch

def graph_reconstruction(x_pred, x, epoch):
  plt.imshow(x_pred.reshape(28, 28), cmap='gray')
  plt.savefig("images/reconstructd_image_epoch_{}.png".format(epoch))
  plt.imshow(x.reshape(28, 28), cmap='gray')
  plt.savefig("images/image_epoch_{}.png".format(epoch))
