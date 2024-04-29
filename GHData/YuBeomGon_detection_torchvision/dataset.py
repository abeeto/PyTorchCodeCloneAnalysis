import os.path
from typing import Any, Callable, Optional, Tuple, List
import os
import torch
import torch.utils.data as data

import numpy as np
import pandas as pd
import albumentations as A
import albumentations.pytorch
import cv2
import math

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
# IMAGE_SIZE = 1200
IMAGE_SIZE= 896

train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),   
    # A.RandomResizedCrop(height=IMAGE_SIZE,width=IMAGE_SIZE,scale=[0.8,1.0],ratio=[0.8,1.2],p=0.8),
#     A.pytorch.ToTensor(), 
], p=1.0, bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8, label_fields=['labels']))    

val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
#     A.pytorch.ToTensor(),     
], p=1.0, bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8, label_fields=['labels']))    

test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
#     A.pytorch.ToTensor(),     
], p=1.0) 


class VisionDataset(data.Dataset):
    """
    Base Class For making datasets which are compatible with torchvision.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.

    Args:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    .. note::

        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        torch._C._log_api_usage_once(f"torchvision.datasets.{self.__class__.__name__}")
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])        

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = np.array(self._load_image(id))
        target = self._load_target(id)
        # print(target)
        
        image_id = [target[0]['image_id']]
        bbox = [target[0]['bbox']]
        
        if len(bbox) > 1 :
            print('target is larger than 1')
            print('box', [target[0]['bbox']])
        
        
        label = [target[0]['category_id']]

        if self.transforms is not None:
            aug = self.transforms(image=image, bboxes=bbox, labels=label)
            image = torch.tensor(aug['image'])
            bbox = aug['bboxes']
            label = aug['labels']
        
        if len(bbox) > 0 :
            # target[0]['bbox'] = bbox[0]
            area = [bbox[0][2] * bbox[0][3]]
#             chagne to xmin,ymin,xmax,ymax
            box = [bbox[0][0], bbox[0][1], bbox[0][2] + bbox[0][0], bbox[0][3]+bbox[0][1]]

            # print(target[0]['bbox'])
        else :
            box = [0,0,1,1]
            area = [0]
            label = 0
            
            
        iscrowd = torch.zeros((1), dtype=torch.int64)
        target = {}
        target['boxes'] = torch.as_tensor([box], dtype=torch.float32)
        target['category_id'] = torch.as_tensor(label, dtype=torch.long) 
        target['labels'] = torch.as_tensor(label, dtype=torch.long) 
        target["image_id"] = torch.as_tensor(image_id, dtype=torch.long)
        target["area"] = torch.as_tensor(area , dtype=torch.float32) 
        target["iscrowd"] = iscrowd 
        
        # print(image.shape)
        
        # image = (image - self.image_mean[None, None, :]) / self.image_std[None, None:, ]
        image = image/255.
        image = image.permute(2,0,1)
            
        return image, target

    def __len__(self) -> int:
        return len(self.ids)