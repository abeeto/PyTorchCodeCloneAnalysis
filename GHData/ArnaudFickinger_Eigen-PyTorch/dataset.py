import torch
import h5py
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tfms, train=True):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.tfms = tfms
        self.train = train
        ds_v_2 = h5py.File(self.data_dir + 'nyu_depth_v2_labeled.mat')
        total_len = len(self.ds_v_2["images"])
        if train:
            self.len = int(0.8*total_len)
            self.img = ds_v_2["images"][:self.len]
            self.disp = ds_v_2["depths"][:self.len]
        else:
            self.len = total_len - int(0.8 * total_len)
            self.img = ds_v_2["images"][int(0.8*total_len):]
            self.disp = ds_v_2["depths"][int(0.8 * total_len):]


    def __getitem__(self, index):

        i = index

        img = np.transpose(self.img[i], axes=[2, 1, 0])
        img = img.astype(np.uint8)

        depth = np.transpose(self.disp[i], axes=[1, 0])
        depth = (depth / depth.max()) * 255
        depth = depth.astype(np.uint8)

        if self.tfms:
            tfmd_sample = self.tfms({"image": img, "depth": depth})
            img, depth = tfmd_sample["image"], tfmd_sample["depth"]
        return (img, depth)

    def __len__(self):
        return self.len