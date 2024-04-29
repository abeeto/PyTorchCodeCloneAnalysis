import os
import sys
import random

import numpy as np
import torch
from albumentations import BboxParams, Compose, HorizontalFlip, LongestMaxSize
from albumentations.pytorch.transforms import ToTensor
from torchvision import transforms


BASE_DIR = sys.path[0]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 将数据增强中框的变化整合，利用albumentation.Compose
# 第一个参数为list类型，所用操作的集合
# 第二个参数为albumentation.BboxParams类型，用于设定boundingbox的格式以及对应的labels列表
def get_aug(aug):
    return Compose(
        aug, bbox_params=BboxParams(format="pascal_voc", label_fields=["gt_labels"])
    )


def prepare(img, boxes, max_dim=None, xflip=False, gt_boxes=None, gt_labels=None):
    aug = get_aug(
        [
            LongestMaxSize(max_size=max_dim),   #在比例不变的前提下对图片缩放
            HorizontalFlip(p=float(xflip)),     #p为概率，dataset类调用是随机为1或0
            ToTensor(
                normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ),
        ]
    )
    augmented = aug(
        image=img, bboxes=boxes, gt_labels=np.full(len(boxes), fill_value=1)#boxes长度的数组，每个元素值均为1,先默认EB提出来的框都是1：bicycle
    )
    augmented_gt = aug(image=img, bboxes=gt_boxes, gt_labels=gt_labels)

    img = augmented["image"].numpy().astype(np.float32)
    boxes = np.asarray(augmented["bboxes"]).astype(np.float32)
    gt_boxes = np.asarray(augmented_gt["bboxes"]).astype(np.float32)

    return img, boxes, gt_boxes



def unique_boxes(boxes, scale=1.0):
    """Returns indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def filter_small_boxes(boxes, min_size):
    """Filters out small boxes."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = (w >= min_size) & (h >= min_size)
    return mask


def swap_axes(boxes):
    """Swaps x and y axes."""
    boxes = boxes.copy()
    boxes = np.stack((boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]), axis=1)
    return boxes

