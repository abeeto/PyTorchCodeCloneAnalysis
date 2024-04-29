import pickle

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from utils import get_file_from_paths


class BaseDataset(Dataset):
    """
    自定义用于MSCOCO数据集的Dataset基类
    """
    def __init__(self, img_dirs, ann_path, transform=None):
        """

        :param img_dirs: images目录路径
        :param ann_path: annotations文件路径
        :param transform: torchvision.transforms定义的图像转换处理
        """
        self.img_dirs = img_dirs
        self.ann_path = ann_path
        self.transform = transform
        self.coco = COCO(ann_path)

    def __getitem__(self, index):
        img_id = self.get_img_id(index)
        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(get_file_from_paths(img_name, self.img_dirs))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, img_id

    def get_img_id(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CocoDataset(BaseDataset):
    """
    用于训练集的Dataset类，读取images和参考captions
    """
    def __init__(self, img_dirs, ann_path, train_pkl, transform=None):
        super(CocoDataset, self).__init__(img_dirs, ann_path, transform)
        self.ann_ids = list(self.coco.anns.keys())

        with open(train_pkl, 'rb') as f:
            self.cap2id = pickle.load(f)

    def __getitem__(self, index):
        img, _ = super(CocoDataset, self).__getitem__(index)
        ann_id = self.ann_ids[index]
        cap = torch.from_numpy(self.cap2id[ann_id]['caption']).long()
        cap_len = self.cap2id[ann_id]['length']
        return img, cap, cap_len

    def get_img_id(self, index):
        """

        :param index: annotation的索引
        :return:
        """
        return self.coco.anns[self.ann_ids[index]]['image_id']

    def __len__(self):
        return len(self.ann_ids)


class ImagesDataset(BaseDataset):
    """
    用于验证集和测试集的Dataset类，只读取images
    """

    def __init__(self, img_dirs, ann_path, transform):
        super(ImagesDataset, self).__init__(img_dirs, ann_path, transform)
        self.img_ids = list(self.coco.imgs.keys())

    def get_img_id(self, index):
        """

        :param index: 图片的索引
        :return:
        """
        return self.img_ids[index]

    def __len__(self):
        return len(self.img_ids)
