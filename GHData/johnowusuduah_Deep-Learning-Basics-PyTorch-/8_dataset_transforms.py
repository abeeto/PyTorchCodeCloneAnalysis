# import dependencies
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


# i. USING CUSTOM TRANSFORMATION CLASS
#**********************************************************************************
class WineDataset(Dataset):

  def __init__(self, transform=None):
    xy = np.loadtxt("./data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
    self.n_samples = xy.shape[0]

    # note that we do not convert to tensor here
    self.x = xy[:, 1:]
    self.y = xy[:, [0]]
    self.transform = transform

  def __getitem__(self, index):
      sample = self.x[index], self.y[index]

      if self.transform:
        sample = self.transform(sample)

      return sample

  def __len__(self):
      return self.n_samples


class ToTensor:
  """
  Torch Tensor Custom Transformation Class
  """
  # this enables classes to be called like functions
  def __call__(self, sample):
    inputs, targets = sample
    return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
  """
  Multiplication (of features) Custom Transformation Class
  """
  def __init__(self, factor):
    self.factor = factor

  def __call__(self, sample):
    inputs, targets = sample
    inputs *= self.factor
    return inputs, targets


# ii. APPLY ToTensor CUSTOM TRANSFORMER
#**********************************************************************************
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))


# iii. APPLY PIPELINE OF TRANSFORMATIONS
#**********************************************************************************
# Pipeline of Transformations
# let's apply a compose transform to see how we can use these two transformations
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))