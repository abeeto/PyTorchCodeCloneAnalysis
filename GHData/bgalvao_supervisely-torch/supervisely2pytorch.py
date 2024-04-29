import sys
import argparse
import os
import json

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch


DATASET_DIR = './supervisely_dataset'


def load_paths(dataset_dir):
    data_paths = {
        'imgs': {},
        'meta': os.path.join(dataset_dir, 'meta.json')
    }

    annotations = []
    images = []
    for dirpath, subdirs, files in os.walk(dataset_dir):
        annotations.extend([
            os.path.join(dirpath, f)
            for f in files if 'ann/' in os.path.join(dirpath, f)
        ])
        images.extend([
            os.path.join(dirpath, f)
            for f in files if 'img/' in os.path.join(dirpath, f)
        ])

    annotations = sorted(annotations)
    images = sorted(images)
    assert len(images) == len(annotations)

    for i, (ann, img) in enumerate(zip(annotations, images)):
        data_paths['imgs'][i] = {
            'img_path': img, 'ann_path': ann
        }

    return data_paths



def parse_annotation(ann_path, classes):

    # print(ann_path)
    with open(ann_path, 'r') as file:
        d = json.load(file)

    # print(json.dumps(d, indent=2))
    boxes = []
    labels = []
    for i, obj in enumerate(d['objects']):
        bbox = obj['points']['exterior']
        label = obj['classTitle']

        xmin = min(bbox[0][0], bbox[1][0])
        xmax = max(bbox[0][0], bbox[1][0])
        ymin = min(bbox[0][1], bbox[1][1])
        ymax = max(bbox[0][1], bbox[1][1])

        if xmin > xmax or ymin > ymax:
            raise Exception('bad coordinates {}\n\n{}'.format(
                    [xmin, ymin, xmax, ymax],
                    json.dumps(d, indent=2)
            ))

        if xmin == xmax or ymin == ymax:
            # in case the bbox is just a straight line
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(classes[label])

    if len(boxes) == 0:
        print('-- criteria left 0 bounding boxes on the image --')

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

    return boxes, labels, areas, iscrowd


def get_num_classes(dataset_root_path=DATASET_DIR):
    path = os.path.join(dataset_root_path, 'meta.json')
    with open(path, 'r') as file:
        d = json.load(file)
    return len(d['classes'])



class SuperviselyDataset(Dataset):

    def __init__(self, root=DATASET_DIR, transforms=None):
        self.root = root
        self.transforms = transforms

        self.data_paths = load_paths(root)
        with open(self.data_paths['meta'], 'r') as file:
            self.classes = {
                c['title'] : i
                for i, c in enumerate(json.load(file)['classes'])
            }


    def __getitem__(self, idx):
        # read image
        img_path = self.data_paths['imgs'][idx]['img_path']
        img = Image.open(img_path).convert('RGB')

        # parse annotation
        ann_path = self.data_paths['imgs'][idx]['ann_path']
        boxes, labels, areas, iscrowd = parse_annotation(ann_path, self.classes)
        img_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        # print(hex(id(self.transforms)))
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


    def __len__(self):
        return len(self.data_paths['imgs'])



if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Convert a supervise.ly-formatted dataset to pytorch's\
        object detection format."
    )

    ap.add_argument(
        "--dataset-dir", type=str, default=DATASET_DIR,
        help="supervisely dataset directory"
    )
    args = ap.parse_args()

    ds = SuperviselyDataset(args.dataset_dir)
    print(ds[0])
