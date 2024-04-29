# -*- coding: utf-8 -*-
"""
@Author ：Shimon-Cheung
@Date   ：2022/4/24 10:34
@Desc   ：pytorch中自定义数据集的使用
"""

import os

from PIL import Image
from torch.utils.data import Dataset


class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        # 进行一些初始化传参的操作
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_list = os.listdir(root_dir + label_dir)

    def __getitem__(self, idx):
        # 返回一条数据，以及label
        img_path = self.root_dir + self.label_dir + "/" + self.img_list[idx]
        img = Image.open(img_path)
        return img, self.label_dir

    def __len__(self):
        # 返回数据集的长度
        return len(self.img_list)


if __name__ == '__main__':
    ants_dataset = Mydata('dataset/hymenoptera_data/train/', 'ants')
    bees_dataset = Mydata('dataset/hymenoptera_data/train/', 'bees')
    train_dataset = ants_dataset + bees_dataset
    print(train_dataset)
