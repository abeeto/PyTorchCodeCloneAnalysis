import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import random
from glob import glob

class MICCAI_Dataset(Dataset):
    def __init__(self, data_root, seq_set, is_train=None, transform=None):
        self.list_ = seq_set
        self.is_train = is_train

        self.dir_root_gt = data_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.resizer = transforms.Compose([transforms.Resize((320, 256))])
        self.xml_dir_list = []

        for i in self.list_:
            dir_sal = self.dir_root_gt + '/seq_' + str(i)
            self.xml_dir_list = self.xml_dir_list + glob(dir_sal + '/xml/*.xml')
            random.shuffle(self.xml_dir_list)

    def __len__(self):
        return len(self.xml_dir_list)

    def __getitem__(self, index):
        _img = Image.open(os.path.dirname(os.path.dirname(self.xml_dir_list[index])) + '/left_frames/' + os.path.basename(
            self.xml_dir_list[index][:-4]) + '.png').convert('RGB')

        _target = Image.open(os.path.dirname(os.path.dirname(self.xml_dir_list[index])) + '/annotations/'
                             + os.path.basename(self.xml_dir_list[index][:-4]) + '.png')

        if self.is_train:
            isAugment = random.random() < 0.5
            if isAugment:
                isHflip = random.random() < 0.5
                if isHflip:
                    _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                    _target = _target.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    _img = _img.transpose(Image.FLIP_TOP_BOTTOM)
                    _target = _target.transpose(Image.FLIP_TOP_BOTTOM)

        _img = np.asarray(_img, np.float32) * 1.0 / 255
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()
        _target = torch.from_numpy(np.array(_target)).long()

        _img = self.resizer(_img)
        _target = self.resizer(_target.unsqueeze(0)).squeeze(0)
        return _img, _target