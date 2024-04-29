from msilib.schema import Directory
import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import random
import xml.etree.ElementTree as ET
class GTInfo:
    img_file: str
    img_shape: tuple
    bbox: dict
    def __init__(self, image, img_file, img_shape, depth, bbox):
        self.image = image
        self.img_file = img_file
        self.img_shape = img_shape
        self.depth = depth
        self.bbox = bbox

    def __str__(self):
        dashline = '-' * 60
        info = f"img_name = {self.img_file}\n"\
        f"img_shape = {int(self.img_shape[0]),int(self.img_shape[1])}\n"\
        f"img_type = {'rgb' if self.depth==3 else 'None'}\n"
        info += f"There are {len(self.bbox)} objects.\n" + dashline + '\n'
        for i, item in enumerate(self.bbox.values()):
            label = item[0]
            cords = item[1]
            info += f'object {i+1}: {label}\n<x1, y1, x2, y2>: {cords}\n'
        return info


class VOCDataset(Dataset):
    def __init__(self, annotation_dir, img_dir, train=True, fraction=0.7, is_random=True):
        self.annotation_dir = annotation_dir
        self.img_dir = img_dir
        self.is_train = train
        self.fraction = fraction

        self.ann_files = self._list_files(is_random)
       
    
    def _list_files(self, is_random=True):
        """Get list of random(is_random) files of specified directory(dirname).
        
        Args:
            dirname (str): The name of directory.
            is_random (bool): Determine whether shuffle files.Default False.
        
        Return:
            (list) : List of file names.
        """
        assert os.path.exists(self.annotation_dir), f"{self.annotation_dir} isn't existed"
        ann_files = os.listdir(self.annotation_dir)
        ann_files = [os.path.splitext(file)[0] for file in ann_files]
        if is_random:
            random.shuffle(ann_files)
        return ann_files
    
    def __len__(self):
        return len(self.ann_files)
    
    def __getitem__(self, idx):
        file_name = self.ann_files[idx]
        gt_info = self._parse_xml_annotation(file_name)
        return gt_info

    def _parse_xml_annotation(self, xml_file_name):
        tree = ET.ElementTree()
        tree.parse(os.path.join(self.annotation_dir, xml_file_name+'.xml'))
        img_name = tree.find('filename').text
        img_shape = tree.find('size')
        width = img_shape.find('width').text
        height = img_shape.find('height').text
        depth = int(img_shape.find('depth').text)
        bbox = {}
        for i, obj in enumerate(tree.findall('object')):
            bbox_label = obj.find('name').text
            bndbox = obj.find('bndbox')

            if int(obj.find('difficult').text) != 0:
                continue
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            bbox[i] = [bbox_label, (xmin, ymin, xmax, ymax)]
            image = Image.open(os.path.join(self.img_dir, img_name))
        return GTInfo(image, img_name, (width, height), depth, bbox)


    
    @property
    def ann_file(self):
        if hasattr(self, 'ann_files'):
            return self.ann_files


def voc_dataset_collate_fn(batch_data):
    batch_bboxes = []
    images = []
    for data in batch_data:
        images.append(np.array(data.image))
        bboxes = []
        for bbox in data.bbox.values():
            bboxes.append(bbox)
        batch_bboxes.append(bboxes)
    return images, batch_bboxes

if __name__ == "__main__":
    train_dataset = VOCDataset('data/VOC2012/Annotations', 'data/VOC2012/JPEGImages')
    train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn = voc_dataset_collate_fn, pin_memory=True)
    batch = next(iter(train_loader))
    print(len(batch[1]))
    print(len(batch[0]))
