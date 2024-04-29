from os.path import join

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class DatasetFromChictopia(data.Dataset):
    """Custom dataset class for loading the pair of images form ChictopiaPlus."""

    mode_list = {'train': {'number': 23012, 'padding': '{0:05d}'},
                 'test': {'number': 2874, 'padding': '{0:04d}'},
                 'val': {'number': 2914, 'padding': '{0:04d}'}}
    real_suffix = '_image:png.png'
    sketch_suffix = '_label_vis:png.png'

    def __init__(self, image_dir, mode: str, transform=None):
        assert mode in ['train', 'test', 'val']
        mode_dict = self.mode_list[mode]
        super().__init__()
        self.image_dir = image_dir
        self.padded_index_list = [mode_dict['padding'].format(index) for index in range(mode_dict['number'])]
        if transform:
            self.transform = transform
        self.transform = transforms.Compose([transforms.Scale(256),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, item):
        sketch_img = Image.open(join(self.image_dir, self.padded_index_list[item] + self.sketch_suffix))
        real_img = Image.open(join(self.image_dir, self.padded_index_list[item] + self.real_suffix))
        sketch_img = self.transform(sketch_img)
        real_img = self.transform(real_img)
        return sketch_img, real_img

    def __len__(self):
        return len(self.padded_index_list)
