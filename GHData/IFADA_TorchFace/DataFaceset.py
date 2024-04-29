import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path,'positive.txt')).readlines())
        self.dataset.extend(open(os.path.join(path,'part.txt')).readlines())
        self.dataset.extend(open(os.path.join(path,'negative.txt')).readlines())

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.path, strs[0])
        cond = torch.Tensor([int(strs[1])])
        offset = torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])])
        img_data1 = np.array(Image.open(img_path))/255.-0.5
        img_data = img_data1.transpose([2,0,1])#NCV
        img_data = torch.Tensor(img_data)
        #img_data = torch.Tensor(np.array(Image.open(img_path))/255.-0.5)
        return img_data,cond,offset
    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = FaceDataset(r'D:\celeba4\12')
    print(dataset[1])



