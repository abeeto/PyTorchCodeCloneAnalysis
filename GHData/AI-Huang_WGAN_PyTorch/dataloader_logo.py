#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-05-22 09:53
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
import glob
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


class LLDIconDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.image_list = glob.glob(os.path.join(root_dir, "*.png"))
        self.image_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        assert idx in range(len(self))
        img_path = self.image_list[idx]
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image


def main():
    root_dir = os.path.expanduser(
        "~/Downloads/LLD-icon_full_data_PNG/LLD_favicons_full_png")
    dataset = LLDIconDataset(root_dir=root_dir)
    image = dataset[0]
    print(image.shape)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    for i, X in enumerate(train_loader):
        print(X.shape)
        if i >= 10:
            break


if __name__ == "__main__":
    main()
