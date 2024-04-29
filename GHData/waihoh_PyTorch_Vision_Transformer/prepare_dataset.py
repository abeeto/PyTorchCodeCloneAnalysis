import torchvision
import torch

# NOTE: cifar-100-python data folder is in content folder
cifar_data = torchvision.datasets.CIFAR100(root='./content/', train=True, download=True)
data_loader = torch.utils.data.DataLoader(cifar_data, batch_size=4, shuffle=True)
