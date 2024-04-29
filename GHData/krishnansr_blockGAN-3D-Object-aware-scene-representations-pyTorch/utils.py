import os
import cv2
import yaml
import glob
import torch
import os.path as osp
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


def load_yaml(config_file):
    config_obj = {}
    if osp.exists(config_file):
        with open(config_file, 'r') as f:
                config_obj = yaml.safe_load(f)
    return config_obj


class CompCarsDataset(Dataset):
    def __init__(self, root, transforms_, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = glob.glob(f'{root}/cars_test/*.jpg') + glob.glob(f'{root}/cars_train/*.jpg')

        if self.__len__() == 0:
            raise FileNotFoundError('Dataset loading error')

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        img = self.transform(Image.open(filepath).convert('RGB'))
        # filename = filepath.split('/')[-1]
        return img

    def __len__(self):
        return len(self.files)


def create_dirs(model_dir):
    if not osp.exists(model_dir):
        os.makedirs(model_dir)


def save_model(filename, epoch, gen, disc=None):
    dd = {
        'gen': gen.state_dict(),
        'epoch': epoch
    }
    if disc is not None:
        dd['disc'] = disc.state_dict()
    torch.save(dd, filename)


def load(filename, gen, map_location=None):
    if map_location is None:
        map_location = torch.device('cpu')

    dd = torch.load(filename, map_location=map_location)
    gen.load_state_dict(dd['gen'])
    print(f'last trained epoch was {dd["epoch"]}')