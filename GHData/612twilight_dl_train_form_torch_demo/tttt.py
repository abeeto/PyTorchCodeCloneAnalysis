# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form
File Name: tttt.py
Author: gaoyw
Create Date: 2020/12/3
-------------------------------------------------
"""

from tqdm import tqdm

from prefetch_generator import BackgroundGenerator
from model.dataloader.mydataloader import Selection_loader
from model.dataloader.mydataset import MultiHeadDataset
from preprocessing.preprocess import Preprocessing


def preprocess_test():
    preprocesser = Preprocessing()
    preprocesser.build_vocab()
    preprocesser.handle_data()


def data_loader_test():
    train_dataset = MultiHeadDataset("prepared_data/train.txt")
    loader = Selection_loader(train_dataset, batch_size=16, num_workers=3)
    pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))
    # pbar = tqdm(enumerate(loader), total=len(loader))
    for _, sample in pbar:
        pass


if __name__ == '__main__':
    data_loader_test()
