from .dataset import Mydatasets
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as Transforms


transforms = Transforms.Compose([Transforms.CenterCrop(20),
                                 Transforms.ToTensor()])
mydataset = Mydatasets(train=True, transform=transforms)
for i,(img,label) in enumerate(mydataset):
    print(i,label)
    pass

train_loader = DataLoader(mydataset, batch_size=1000, shuffle=True)
print(len(train_loader))