import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random


class DataSet1(Dataset):
    def __init__(self, root, mode, transforms_=None):
        self.transform = transforms_
        self.fnames = sorted(glob.glob(root + '/*.bmp'))
        random.shuffle(self.fnames)
        if mode == 'Train':
            self.fnames = self.fnames[:-10]
        elif mode == 'Validation':
            self.fnames = self.fnames[-10:]
        elif mode == 'Testing':
            self.fnames = self.fnames

        # print()

    def __getitem__(self, index):
        f = self.fnames[index]
        img = self.transform(Image.open(f).convert('YCbCr'))
        # debug_img = transforms.functional.to_pil_image(img[0, :, :])
        # debug_img.show()
        return img[0, :, :].unsqueeze(0)

    def __len__(self):
        return len(self.fnames)
