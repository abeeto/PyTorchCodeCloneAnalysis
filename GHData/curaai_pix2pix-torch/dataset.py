import os 
import numpy as np
import torch 
import torch.utils.data as data
from PIL import Image
from torchvision import datasets, transforms


class Dataset(data.Dataset):
    def __init__(self, src_path, trg_path, transform):
        self.src_path = src_path
        self.trg_path = trg_path

        self.src_images = os.listdir(src_path)
        self.trg_images = os.listdir(trg_path)
        self.transform = transform


    def __getitem__(self, index):
        src_img = Image.open(os.path.join(self.src_path, self.src_images[index]))
        src_img = src_img.convert('RGB')
        trg_img = Image.open(os.path.join(self.trg_path, self.trg_images[index]))
        trg_img = trg_img.convert('RGB')

        if self.transform is not None:
            src_img = self.transform(src_img)
            trg_img = self.transform(trg_img)

        return src_img, trg_img

    def __len__(self):
        return len(self.src_images)


def get_loader(batch_size, src_path, trg_path, transform, shuffle=True):
    dataset = Dataset(src_path, trg_path, transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return dataloader

if __name__ == '__main__':
    a = torch.cuda.device(0)
    b = torch.cuda.device(1)

    pass
    