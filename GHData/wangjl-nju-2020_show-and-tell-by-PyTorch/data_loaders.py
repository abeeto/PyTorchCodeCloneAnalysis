import torch
import numpy as np

from torch.utils.data import DataLoader
from coco_dataset import CocoDataset
from coco_dataset import ImagesDataset
from torchvision.transforms import transforms
from PIL import Image


def train_collate_fn(data):
    # 对caption按长度降序排列
    data.sort(key=lambda x: x[2], reverse=True)
    img, cap, cap_len = zip(*data)
    img = torch.stack(img, 0)
    cap = torch.stack(cap, 0)
    return img, cap, np.array(cap_len)


def get_train_transforms():
    """
    定义训练集图片转换

    :return:
    """
    return transforms.Compose([transforms.Resize([300, 300], Image.ANTIALIAS),
                               transforms.RandomCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))])


def get_val_transforms():
    """
    定义验证集图片转换

    :return:
    """
    return transforms.Compose([transforms.Resize([224, 224], Image.ANTIALIAS),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))])


def get_test_transforms():
    """
    定义测试集图片转换

    :return:
    """
    return get_val_transforms()


def get_train_loader(hparams,
                     collate_fn=train_collate_fn,
                     transform=get_train_transforms()):
    """
    训练集Dataloader实例

    :param hparams:
    :param collate_fn:
    :param transform:
    :return:
    """
    return DataLoader(dataset=CocoDataset(img_dirs=hparams.img_dirs,
                                          ann_path=hparams.train_cap,
                                          train_pkl=hparams.train_pkl,
                                          transform=transform),
                      batch_size=hparams.batch_size,
                      shuffle=True,
                      num_workers=hparams.num_workers,
                      collate_fn=collate_fn,
                      drop_last=True)


def get_val_loader(hparams, transform=get_val_transforms()):
    """
    验证集DataLoader实例

    :param hparams:
    :param transform:
    :return:
    """
    return DataLoader(dataset=ImagesDataset(img_dirs=hparams.img_dirs,
                                            ann_path=hparams.val_cap,
                                            transform=transform),
                      batch_size=1,
                      shuffle=False,
                      num_workers=1)


def get_test_loader(hparams, transform=get_test_transforms()):
    """
    测试集DataLoader实例

    :param hparams:
    :param transform:
    :return:
    """
    return DataLoader(dataset=ImagesDataset(img_dirs=hparams.img_dirs,
                                            ann_path=hparams.test_cap,
                                            transform=transform),
                      batch_size=1,
                      shuffle=False,
                      num_workers=1)
