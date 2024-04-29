# -*- coding: utf-8 -*-
from torchvision import transforms
import torch.utils.data as data
import torch
from config import *
import os
from PIL import Image
import random


class Fashion_attr_prediction(data.Dataset):
    def __init__(self, type="train", transform = None, img_path=None,  crop=True):
        self.transform = transform
        # self.target_transform = target_transform
        self.crop = crop
        # type_all = ["train", "test", "all", "triplet", "single"]
        self.type = type
        if type == "single":
            self.img_path = img_path
            return
        self.train_list = []
        self.val_list = []
        # self.train_dict = {i: [] for i in range(CATEGORIES)}
        self.test_list = []
        self.all_list = []
        self.bbox = dict()
        self.attr = dict()
        self.anno = dict()
        self.transform_attr()
        # self.read_digit_lines()
        # self.read_char_lines()
        # self.read_crop()
        # self.read_partition_category()
        # self.read_bbox()

    def __len__(self):
        if self.type == "all":
            return len(self.all_list)
        elif self.type == "train":
            return len(self.train_list)
        elif self.type == "val":
            return len(self.val_list)
        elif self.type == "test":
            return len(self.test_list)
        else:
            return 1
    def transform_attr(self):
        #img files
        train_name_file = os.path.join(DATASET_BASE, r'split', r'train.txt')
        val_name_file = os.path.join(DATASET_BASE, r'split', r'val.txt')
        test_name_file = os.path.join(DATASET_BASE, r'split', r'test.txt')
        #y label files
        train_attr_file = os.path.join(DATASET_BASE, r'split', r'train_attr.txt')
        val_attr_file = os.path.join(DATASET_BASE, r'split', r'val_attr.txt')

        train_bbox_file = os.path.join(DATASET_BASE, r'split', r'train_bbox.txt')
        val_bbox_file = os.path.join(DATASET_BASE, r'split', r'val_bbox.txt')
        test_bbox_file = os.path.join(DATASET_BASE, r'split', r'test_bbox.txt')
        #get a list of lists of image names
        train_name = self.read_char_lines(train_name_file)
        val_name = self.read_char_lines(val_name_file)
        self.train_list = self.train_list + train_name
        self.val_list = self.val_list +val_name
        test_name = self.read_char_lines(test_name_file)
        self.test_list = self.test_list + test_name
        self.all_list = self.train_list + self.val_list + self.test_list
        #bbox files are read the same way as attr files
        train_bbox = self.read_digit_lines(train_bbox_file)
        val_bbox = self.read_digit_lines(val_bbox_file)
        test_bbox = self.read_digit_lines(test_bbox_file)

        train_attr = self.read_digit_lines(train_attr_file)
        val_attr = self.read_digit_lines(val_attr_file)

        for i in range (len(train_name)):
            name = train_name[i][0]
            tmp_attr = train_attr[i]
            tmp_attr = torch.tensor(tmp_attr)
            # print ("tmp atr", tmp_attr)
            self.bbox[name] = train_bbox[i]
            # attr = [0] * 26
            # attr[tmp_attr[0]] = 1
            # attr[7 + tmp_attr[1]] = 1
            # attr[10 + tmp_attr[2]] = 1
            # attr[13 + tmp_attr[3]] = 1
            # attr[17 + tmp_attr[4]] = 1
            # attr[23 + tmp_attr[5]] = 1
            # attr = torch.tensor(attr)
            # self.attr[name] = attr
            self.attr[name] = tmp_attr
        for i in range (len(val_name)):
            name = val_name[i][0]
            tmp_attr = val_attr[i]

            tmp_attr = torch.tensor(tmp_attr)
            # print (tmp_attr.shape)
            self.bbox[name] = val_bbox[i]
            # attr = [0] * 26
            # attr[tmp_attr[0]] = 1
            # attr[7 + tmp_attr[1]] = 1
            # attr[10 + tmp_attr[2]] = 1
            # attr[13 + tmp_attr[3]] = 1
            # attr[17 + tmp_attr[4]] = 1
            # attr[23 + tmp_attr[5]] = 1
            # attr = torch.tensor(attr)
            # self.attr[name] = attr
            self.attr[name] = tmp_attr
        for i in range (len(test_name)):
            name = test_name[i][0]
            self.bbox[name] = test_bbox[i]


    def read_digit_lines(self, path):
        with open(path) as file:
            with open(path) as file:
                lines = file.readlines()
                lines = list(filter(lambda x: len(x) > 0, lines))
                lines = list(map(lambda x: (x.strip().split()), lines))
                for i in range(len(lines)):
                    lines[i] = list(map(int, lines[i]))
            return lines

    def read_char_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def read_crop(self, img_path):
        img_full_path = os.path.join(DATASET_BASE, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.crop:
            x1, y1, x2, y2 = self.bbox[img_path]
            if x1 < x2 <= img.size[0] and y1 < y2 <= img.size[1]:
                img = img.crop((x1, y1, x2, y2))
        return img

    def __getitem__(self, index):


        if self.type == "single":
            img_path = self.img_path
            img = self.read_crop(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img,
        if self.type == "all":
            img_path = self.all_list[index][0]
        elif self.type == "train" :
            img_path = self.train_list[index][0]
            attr = self.attr[img_path]
        elif self.type == "val":
            img_path = self.val_list[index][0]
            attr = self.attr[img_path]

        else:
            img_path = self.test_list[index][0]
        # target = self.attr[img_path]
        img = self.read_crop(img_path)


        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, attr if self.type == "train" or self.type == "val" else img_path
