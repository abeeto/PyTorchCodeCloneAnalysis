from torch.utils.data import Dataset, DataLoader
import pandas as pd
import medpy.io as mpy
import numpy as np
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.ndimage import gaussian_filter
import elasticdeform
import config
import torchio as tio
import torch
import os
import nibabel as niib

class BrainLoader(Dataset):

    def __init__(self, dataset, train=True):
        self.df = pd.read_csv(dataset)
        self.df = self.df
        self.train = train
        self.traintransforms = tio.Compose(
            [tio.Resize(target_shape=(config.SIZE, config.SIZE, config.SIZE), image_interpolation='linear'),
             tio.OneOf({
                 tio.RandomAffine(degrees=(180, 180, 180)): 0.8,
                 tio.RandomElasticDeformation(max_displacement=15): 0.2,},
                 p=0.75,),
             tio.RandomFlip(axes=(0), p=0.5),
             tio.RandomFlip(axes=(1), p=0.5),
             tio.RandomFlip(axes=(2), p=0.5),
             ]
        )

        self.testtransforms = tio.Compose(
            [tio.Resize(target_shape=(config.SIZE, config.SIZE, config.SIZE), image_interpolation='linear'),
             ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.read_image(self.df['input'][idx])
        gt = self.read_gt(self.df['gt'][idx])

        img, gt = self.padding(img, gt)
        mask = gt != 3
        gt = gt * mask
        mask = gt == 4
        gt[mask] = gt[mask] - 1

        img, gt = torch.tensor(img).float(), torch.tensor(gt[np.newaxis, ...]).float()
        if self.train:
            subject = tio.Subject(img=tio.ScalarImage(tensor=img),
                                  gt=tio.LabelMap(tensor=gt))
            transform = self.traintransforms(subject)
            img, gt = transform['img'][tio.DATA], transform['gt'][tio.DATA]
            gt = gt.squeeze()
        else:
            subject = tio.Subject(img=tio.ScalarImage(tensor=img),
                                  gt=tio.LabelMap(tensor=gt))
            transform = self.testtransforms(subject)
            img, gt = transform['img'][tio.DATA], transform['gt'][tio.DATA]
            gt = gt.squeeze()
        gttrans = np.zeros([3, config.SIZE, config.SIZE, config.SIZE])
        wt_mask1 = gt == 1
        wt_mask2 = gt == 3
        wt_mask = wt_mask1 + wt_mask2
        gttrans[0][wt_mask] = 1
        et_mask = gt == 3
        gttrans[1][et_mask] = 1
        ed_mask = gt == 2
        gttrans[2][ed_mask] = 1

        img = self.normalize(img)
        return img, gttrans

    def read_gt(self, file):
        file = niib.load(file)
        data = file.get_data()
        return data

    def read_image(self, dir):
        multlimod_data = np.zeros([4, 240, 240, 155])
        for exp in os.listdir(dir):
            index=-1
            if '_seg' in exp:
                continue
            if '_t2' in exp:
                index = 0
            if '_t1' in exp:
                index = 1
            if '_t1ce' in exp:
                index = 2
            if '_flair' in exp:
                index = 3
            file = niib.load(dir + '/' + exp)
            data = file.get_data()
            multlimod_data[index, ...] = data
        return multlimod_data

    def normalize(self, x):
        mean = torch.mean(x, dim=[1, 2, 3])
        std = torch.std(x, dim=[1, 2, 3])
        mean = torch.reshape(mean, [4, 1, 1, 1])
        std = torch.reshape(std, [4, 1, 1, 1])
        x = (x - mean) / std
        return x

    def padding(self, x, y):
        channels, x_dim, y_dim, z_dim = x.shape
        diff = x_dim - z_dim
        upper = diff // 2
        lower = diff - upper
        padded_x = np.zeros([channels, x_dim, x_dim, x_dim])
        padded_y = np.zeros([x_dim, x_dim, x_dim])
        padded_x[..., upper:-lower] = x
        padded_y[..., upper:-lower] = y
        return padded_x, padded_y

    def rotation3d(self, x, y):
        alpha, beta, gamma = np.random.uniform(0, 360, [3])
        x = rotate(x, angle=gamma, axes=(0, 1), mode='nearest', reshape=False)
        x = rotate(x, angle=beta, axes=(0, 2), mode='nearest', reshape=False)
        x = rotate(x, angle=alpha, axes=(1, 2), mode='nearest', reshape=False)

        y = rotate(y, angle=gamma, axes=(0, 1), mode='nearest', order=0, reshape=False)
        y = rotate(y, angle=beta, axes=(0, 2), mode='nearest', order=0, reshape=False)
        y = rotate(y, angle=alpha, axes=(1, 2), mode='nearest', order=0, reshape=False)
        return x, y

    def random_brightness_contrast(self, x):
        alpha = np.random.uniform(0.7, 1.4, 1)
        beta = np.random.uniform(0.0, 2, 1)
        x = x * alpha + beta
        return x

if __name__ == '__main__':
    from vedo.applications import SlicerPlotter
    from vedo import *
    csv_path = '/media/tensorist/Extreme SSD/brats2020/trainset.csv'

    loader = BrainLoader(csv_path)
    weights = {'0': 0, '1': 0, '2': 0, '3': 0, 'num': 0}
    for i in range(len(loader)):
        img, gt = loader[i]
        img = img.squeeze()

        gt_true = torch.zeros([128, 128, 128])
        wt_mask = gt[0] == 1
        gt_true[wt_mask] = 1
        et_mask = gt[1] == 1
        gt_true[et_mask] = 2
        ed_mask = gt[2] == 1
        gt_true[ed_mask] = 3

        vol = Volume(img[0])

        plt = SlicerPlotter(vol,
                            bg='white', bg2='lightblue',
                            cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
                            useSlider3D=False,
                            )

        plt.show().close()




