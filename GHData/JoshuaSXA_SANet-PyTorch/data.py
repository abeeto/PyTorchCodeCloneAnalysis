import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
from generate_dataset import get_data_from_file


def img_padding(img_arr, gt_arr):
    (h, w, c) = img_arr.shape
    max_len = h if w <= h else w
    target_size = (max_len // 8 + 1) * 8 if max_len % 8 else max_len

    padding_top = (target_size - h) // 2
    padding_bottom = target_size - padding_top - h
    padding_left = (target_size - w) // 2
    padding_right = target_size - padding_left - w
    img_arr = np.pad(img_arr, ((padding_top, padding_bottom), (padding_left, padding_right), (0,0)), 'constant', constant_values=0)
    gt_arr = np.pad(gt_arr, ((padding_top, padding_bottom), (padding_left, padding_right), (0,0)), 'constant', constant_values=0)
    img_arr = img_arr.transpose(2, 0, 1)
    gt_arr = gt_arr.transpose(2, 0, 1)
    return img_arr, gt_arr



class MyDataset(Dataset):

    def __init__(self, img_file_path, label_file_path, num_workers=4):
        self.img_paths = sorted(glob(os.path.join(img_file_path, '*.png')))
        self.label_paths = sorted(glob(os.path.join(label_file_path, '*.txt')))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.img_paths[index]
        gt_path = self.label_paths[index]
        img_arr, gt_arr = get_data_from_file(img_path, gt_path)
        img_arr, gt_arr = img_padding(img_arr, gt_arr)
        return torch.from_numpy(img_arr), torch.from_numpy(gt_arr)

# img_file_path = os.path.join('StoneData', 'images')
# gt_file_path = os.path.join('StoneData', 'ground_truth')
# dataset = MyDataset(img_file_path, gt_file_path)
# for (img, gt) in dataset:
#     print(img.shape, gt.shape)
