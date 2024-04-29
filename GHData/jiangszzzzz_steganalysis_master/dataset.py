from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip
)
from glob import glob
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import gc
from torch.utils.data import Dataset


class Alaska2Dataset(Dataset):

    def __init__(self, df, augmentations=None):
        self.data = df
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label = self.data.loc[idx]
        im = cv2.imread(fn)[:, :, ::-1]
        if self.augment:
            im = self.augment(image=im)
        return im, label


if __name__ == "__main__":

    data_dir = '/steganalysis/dataset'
    sample_size = 50000
    val_size = int(sample_size * 0.25)
    train_fn, val_fn = [], []
    train_labels, val_labels = [], []

    folder_names = ['Cover_jpg/', 'JMiPOD/', 'JUNIWARD/', 'UERD/']  # label 1 2 3

    for label, folder in enumerate(folder_names):
        # python的glob模块可以对文件夹下所有文件进行遍历，并保存为一个list列表
        train_filenames = glob(f"{data_dir}/{folder}/*.jpg")[:sample_size]
        np.random.shuffle(train_filenames)
        # train
        train_fn.extend(train_filenames[val_size:])
        train_labels.extend(np.zeros(len(train_filenames[val_size:], )) + label)
        # val
        val_fn.extend(train_filenames[:val_size])
        val_labels.extend(np.zeros(len(train_filenames[:val_size], )) + label)

    assert len(train_labels) == len(train_fn), "wrong labels"
    assert len(val_labels) == len(val_fn), "wrong labels"

    # 验证
    train_df = pd.DataFrame({'ImageFileName': train_fn, 'Label': train_labels}, columns=['ImageFileName', 'Label'])
    train_df['Label'] = train_df['Label'].astype(int)
    val_df = pd.DataFrame({'ImageFileName': val_fn, 'Label': val_labels}, columns=['ImageFileName', 'Label'])
    val_df['Label'] = val_df['Label'].astype(int)

    print(train_df)
    train_df.Label.hist()

    # 测试 class
    img_size = 512
    AUGMENTATIONS_TRAIN = Compose([
        Resize(img_size, img_size, p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        JpegCompression(quality_lower=75, quality_upper=100, p=0.5),
        ToFloat(max_value=255),
        ToTensorV2()
    ], p=1)
    AUGMENTATIONS_TEST = Compose([
        Resize(img_size, img_size, p=1),
        ToFloat(max_value=255),
        ToTensorV2()
    ], p=1)

    temp_df = train_df.sample(64).reset_index(drop=True)
    train_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)
    batch_size = 64
    num_workers = 0

    temp_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers, shuffle=False)

    images, labels = next(iter(temp_loader))
    images = images['image'].permute(0, 2, 3, 1)
    max_images = 64
    grid_width = 16
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width + 1, grid_height + 1))

    for i, (im, label) in enumerate(zip(images, labels)):
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(im.squeeze())
        ax.set_title(str(label.item()))
        ax.axis('off')

    plt.suptitle("0: COVER, 1: JMiPOD, 2: JUNIWARD, 3:UERD")
    plt.show()
    del images
    gc.collect()

    # 测试数据集 图像转置

    train_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TRAIN)
    batch_size = 64
    num_workers = 0

    temp_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers, shuffle=False)

    images, labels = next(iter(temp_loader))
    images = images['image'].permute(0, 2, 3, 1)
    max_images = 64
    grid_width = 16
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width,
                            figsize=(grid_width + 1, grid_height + 1))

    for i, (im, label) in enumerate(zip(images, labels)):
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(im.squeeze())
        ax.set_title(str(label.item()))
        ax.axis('off')

    plt.suptitle("0: No Hidden Message, 1: JMiPOD, 2: JUNIWARD, 3:UERD")
    plt.show()
    del images, temp_df
    gc.collect()
