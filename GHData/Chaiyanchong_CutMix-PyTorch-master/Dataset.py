import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
class MyDataset(Dataset):
    def __init__(self, root, datatxt, transform=None):
        super(MyDataset, self).__init__()
        fh = open(root+datatxt,'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn,label = self.imgs[index]
        img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)