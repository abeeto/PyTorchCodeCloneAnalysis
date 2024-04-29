import platform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as f
from pycocotools.coco import COCO
import os
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import random
from PIL import Image
from tqdm import tqdm
import json
import xml.etree.ElementTree as et
from BoundingBox import *
import numpy as np
from xml.dom.minidom import Document

KITTI_file = r'D:\DateSet\KITTI'
Traffic_file = r'D:\DateSet\Traffic'
COCO_file = r'D:\DateSet\COCO'
VOC_file = r'D:\DateSet\VOC'


class VOCDetection(Dataset):
    """
    Pascal VOC数据集，可以用在多种CV任务中，本类只适用于目标检测任务
    VOC数据集包含20个目标类别，训练与验证图片共计一万余张
    """

    def __init__(self, root_file=VOC_file, mode="train",
                 transforms_image=None, transforms_label=None, transforms_all=None, make_labels=False,
                 dataset='VOC'):
        """
        :param root_file: 数据集的根目录
        :param mode: 训练集"train"或验证集"val"
        :param transforms_image: 对输入的样本进行变换的方法
        :param transforms_label: 对输入的标签进行变换的方法
        :param transforms_all: 对输入的样本与标签都进行变换的方法
        :param make_labels: 是否在加载数据阶段制作最终标签
        :param dataset: 数据集，默认为VOC07+12，否则加载指定文件夹
        """
        super(VOCDetection, self).__init__()
        assert mode in ["train", "val", "trainval", "test"], \
            'dataset must be in ["train", "val", "trainval", "test"]'
        self.dataset = dataset
        self.transfer = None
        if dataset == 'VOC':
            if mode != "test":
                # 检查相关文件是否存在
                assert os.path.exists(root_file), "file '{}' does not exist.".format(root_file)
                self.img_root = [os.path.join(root_file, "VOC2007", "JPEGImages"),
                                 os.path.join(root_file, "VOC2012", "JPEGImages")]  # 图片文件根目录
                assert os.path.exists(self.img_root[0]), "path '{}' does not exist.".format(self.img_root[0])
                assert os.path.exists(self.img_root[1]), "path '{}' does not exist.".format(self.img_root[1])
                self.anno_root = [os.path.join(root_file, "VOC2007", "Annotations"),
                                  os.path.join(root_file, "VOC2012", "Annotations")]  # 标签文件根目录
                assert os.path.exists(self.anno_root[0]), "path '{}' does not exist.".format(self.anno_root[0])
                assert os.path.exists(self.anno_root[1]), "path '{}' does not exist.".format(self.anno_root[1])
                self.image_names_file = [os.path.join(root_file, "VOC2007", "ImageSets", "Main", mode + '.txt'),
                                         os.path.join(root_file, "VOC2012", "ImageSets", "Main",
                                                      mode + '.txt')]  # 图片名称文件
                assert os.path.exists(self.image_names_file[0]), "file '{}' does not exist.".format(
                    self.image_names_file[0])
                assert os.path.exists(self.image_names_file[1]), "file '{}' does not exist.".format(
                    self.image_names_file[1])

                # 读取图片名称
                self.image_ids = [name.strip() for name in open(self.image_names_file[0], 'r').readlines()] + \
                                 [name.strip() for name in open(self.image_names_file[1], 'r').readlines()]

            else:
                self.img_root = [os.path.join(root_file, "VOCtest", "JPEGImages")]
                assert os.path.exists(self.img_root[0]), "path '{}' does not exist.".format(self.img_root[0])
                self.anno_root = [os.path.join(root_file, "VOCtest", "Annotations")]  # 标签文件根目录
                assert os.path.exists(self.anno_root[0]), "path '{}' does not exist.".format(self.anno_root[0])
                self.image_names_file = [
                    os.path.join(root_file, "VOCtest", "ImageSets", "Main", mode + '.txt')]  # 图片名称文件
                assert os.path.exists(self.image_names_file[0]), "file '{}' does not exist.".format(
                    self.image_names_file[0])

                # 读取图片名称
                self.image_ids = [name.strip() for name in open(self.image_names_file[0], 'r').readlines()]

            # VOC类别包含以下4大类（人、交通工具、家具、动物），共20小类
            self.classes = ['person',
                            'car', 'bus', 'bicycle', 'motorbike', 'aeroplane', 'boat', 'train',
                            'chair', 'sofa', 'diningtable', 'tvmonitor', 'bottle', 'pottedplant',
                            'cat', 'dog', 'cow', 'horse', 'sheep', 'bird']

        elif dataset == 'KITTI':
            # 加载指定数据集
            self.img_root = [os.path.join(root_file, dataset, "JPEGImages")]
            assert os.path.exists(self.img_root[0]), "path '{}' does not exist.".format(self.img_root[0])
            self.anno_root = [os.path.join(root_file, dataset, "Annotations")]  # 标签文件根目录
            assert os.path.exists(self.anno_root[0]), "path '{}' does not exist.".format(self.anno_root[0])
            self.image_names_file = [os.path.join(root_file, dataset, "ImageSets", "Main", mode + '.txt')]  # 图片名称文件
            assert os.path.exists(self.image_names_file[0]), "file '{}' does not exist.".format(
                self.image_names_file[0])

            # 读取图片名称
            self.image_ids = [name.strip() for name in open(self.image_names_file[0], 'r').readlines()]
            self.transfer = json.load(open(os.path.join(root_file, dataset, 'transfer.json'), 'r'))
            self.classes = [cls.strip() for cls in
                            open(os.path.join(root_file, dataset, 'labels.txt'), 'r').readlines()]

        else:
            raise ValueError

        self.mode = mode
        self.transforms_image = transforms_image
        self.transforms_label = transforms_label
        self.transforms_all = transforms_all
        self.make_labels = make_labels

        print(f'{mode} dataset of {len(self.image_ids)} images in total')

    def ParseTargets(self, img_id):
        """
        解析标签xml文件
        :param img_id: 图片名称
        :return: 目标字典{"anchors": anchors, "image_id": [img_id]}，anchors对应目标真实值矩阵，image_id对应图片名称
        """
        # 读取xml文件，获得根节点
        targets_xml = et.parse(os.path.join(self.anno_root[img_id[0] == '2'], img_id + '.xml'))
        root = targets_xml.getroot()

        # 获取图片尺寸
        size = root.find("size")
        w, h = float(size.find("width").text), float(size.find("height").text)

        # 获取图片包含的目标坐标、类别信息
        boxes, classes = [], []
        for ob in root.findall("object"):
            cls = ob.find("name").text
            if self.transfer is not None:
                cls = self.transfer[cls]
                if cls == 'Misc':
                    continue
                if cls == 'DontCare' and self.mode != 'val':
                    continue
            classes.append(self.classes.index(cls) + 1)
            bbox = ob.find("bndbox")
            boxes.append([float(bbox.find("xmin").text), float(bbox.find("ymin").text),
                          float(bbox.find("xmax").text), float(bbox.find("ymax").text)])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.as_tensor(classes, dtype=torch.float32).unsqueeze(1)

        # 坐标归一化
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        boxes.clamp_(min=0, max=1)

        # 组装成结果返回
        anchors = torch.cat((boxes, classes), dim=1)

        target = {"anchors": anchors, "image_id": img_id}
        return target

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        # 读取图片
        img = Image.open(os.path.join(self.img_root[img_id[0] == '2'], img_id + '.jpg')).convert('RGB')

        # 解析目标值格式
        target = self.ParseTargets(img_id)
        if self.transforms_image is not None:
            img = self.transforms_image(img)
        if self.transforms_all is not None:
            img, target = self.transforms_all(img, target)
        target["anchors"] = target["anchors"][torchvision.ops.remove_small_boxes(target["anchors"], min_size=0.003)]
        if target["anchors"].shape[0] == 0:
            target["anchors"] = torch.zeros((1, 5))  # 防止没有目标导致tensor为空
        if self.make_labels and self.transforms_label is not None:
            target["anchors"] = self.transforms_label(target["anchors"].unsqueeze(0))
        return img, target

    def __len__(self):
        return len(self.image_ids)

    def GetHW(self, index=None, img_id=None):
        """
        获取图像宽和高
        :param index: 图片索引
        :param img_id: 图像编号
        :return:
        """
        if index is not None:
            img_id = self.image_ids[index]
        # 读取xml文件，获取图像尺寸
        root = et.parse(os.path.join(self.anno_root[img_id[0] == '2'], img_id + '.xml')).getroot()
        size = root.find("size")
        w, h = float(size.find("width").text), float(size.find("height").text)
        return h, w

    def collate_with_match(self, batch):
        """
        组装样本时匹配标签
        """
        features, targets = zip(*batch)

        # 图片打包成tensor: [batch_size, channels, h, w]
        features = torch.stack(features)

        # 重新整合目标值格式
        anchors = [target['anchors'] for target in targets]  # 目标锚框放在一起
        image_ids = [target['image_id'] for target in targets]  # 每个图片的id放一起

        # 由于每个图片包含的目标数量不相等，为了组合成一个矩阵，需要将所有图片的目标数量padding成相同的数量（以最多的那个为基准）
        max_object_num = max([anchor.shape[0] for anchor in anchors])
        for i in range(len(anchors)):
            pad = (0, 0, 0, max_object_num - anchors[i].shape[0])  # 行的末尾padding到最大行数
            anchors[i] = f.pad(anchors[i], pad, 'constant', 0)
        anchors = torch.stack(anchors)

        # 变换目标值格式
        anchors = self.transforms_label(anchors)

        targets = {'anchors': anchors, 'image_ids': image_ids}

        return features, targets

    @staticmethod
    def collate_matched(batch):
        """
        组装已匹配标签的样本
        """
        features, targets = zip(*batch)

        # 图片打包成tensor: [batch_size, channels, h, w]
        features = torch.stack(features)

        # 重新整合目标值格式
        image_ids = [target['image_id'] for target in targets]
        offset_targets = [target['anchors'][0] for target in targets]
        classes_targets = [target['anchors'][1] for target in targets]
        priors_mask = [target['anchors'][2] for target in targets]

        offset_targets = torch.cat(offset_targets, dim=0)
        classes_targets = torch.cat(classes_targets, dim=0)
        priors_mask = torch.cat(priors_mask, dim=0)

        targets = {'anchors': (offset_targets, classes_targets, priors_mask), 'image_ids': image_ids}

        return features, targets

    @staticmethod
    def collate_fn(batch):
        """
        组装未处理样本
        """
        features, targets = zip(*batch)

        # 图片打包成tensor: [batch_size, channels, h, w]
        features = torch.stack(features)

        # 重新整合目标值格式
        anchors = [target['anchors'] for target in targets]  # 目标锚框放在一起
        image_ids = [target['image_id'] for target in targets]  # 每个图片的id放一起

        # 由于每个图片包含的目标数量不相等，为了组合成一个矩阵，需要将所有图片的目标数量padding成相同的数量（以最多的那个为基准）
        max_object_num = max([anchor.shape[0] for anchor in anchors])
        for i in range(len(anchors)):
            pad = (0, 0, 0, max_object_num - anchors[i].shape[0])  # 行的末尾padding到最大行数
            anchors[i] = f.pad(anchors[i], pad, 'constant', 0)

        targets = {'anchors': torch.stack(anchors), 'image_ids': image_ids}

        return features, targets

    def ShowImage(self, image_id, image_transforms=None, all_transforms=None, show_score=True):
        img = torchvision.io.read_image(
            os.path.join(self.img_root[image_id[0] == '2'], image_id + '.jpg'))
        target = self.ParseTargets(image_id)
        target['anchors'][:, -1] -= 1
        if image_transforms is not None:
            img = image_transforms(img)
        if all_transforms is not None:
            img, target = all_transforms(img, target)
        img = img.permute(1, 2, 0)
        target = target['anchors']
        target = torch.cat((target, torch.ones((target.shape[0], 1))), dim=1)
        display(img, target, 0.5, classes=self.classes, show_score=show_score)

    def GetAllTargets(self):
        targets = []
        for img_id in tqdm(self.image_ids, desc='parsing'):
            targets.append(self.ParseTargets(img_id)["anchors"])
        return torch.stack(targets)


class CocoDetection(Dataset):
    """
    COCO2017目标检测数据集，COCO数据集可以用在多种任务，本类只适用于目标检测任务
    COCO数据集包含80个目标类别，用于目标检测的部分包含11万余张训练图片与5000张验证图片
    """

    def __init__(self, root_file=COCO_file, mode="train",
                 transforms_image=None, transforms_label=None, transforms_all=None, make_labels=False):
        """
        :param root_file: 数据集的根目录
        :param mode: 训练集"train"或验证集"val"
        :param transforms_image: 对输入的样本进行变换的方法
        :param transforms_label: 对输入的标签进行变换的方法
        :param transforms_all: 对输入的样本与标签都进行变换的方法
        :param make_labels: 是否在加载数据阶段制作最终标签
        """
        super(CocoDetection, self).__init__()
        assert mode in ["train", "val"], 'dataset must be in ["train", "val"]'
        assert os.path.exists(root_file), "file '{}' does not exist.".format(root_file)
        self.img_root = os.path.join(root_file, "{}2017".format(mode))
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        anno_file = "instances_{}2017.json".format(mode)
        self.anno_path = os.path.join(root_file, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.mode = mode
        self.transforms = transforms
        self.coco = COCO(self.anno_path)

        if mode == "train":
            # 获取coco数据索引与类别名称的关系
            # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
            coco_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])

            # 将stuff91的类别索引重新编排，从1到80
            coco91to80 = dict([(str(k), idx + 1) for idx, (k, _) in enumerate(coco_classes.items())])
            json_str = json.dumps(coco91to80, indent=4)
            with open('coco91_to_80.json', 'w') as json_file:
                json_file.write(json_str)

            # 记录重新编排后的索引以及类别名称关系
            coco80_info = dict([(str(idx + 1), v) for idx, (_, v) in enumerate(coco_classes.items())])
            json_str = json.dumps(coco80_info, indent=4)
            with open('coco80_indices.json', 'w') as json_file:
                json_file.write(json_str)
        else:
            # 如果是验证集就直接读取生成好的数据
            coco91to80_path = 'coco91_to_80.json'
            assert os.path.exists(coco91to80_path), "file '{}' does not exist.".format(coco91to80_path)

            coco91to80 = json.load(open(coco91to80_path, "r"))

        self.coco91to80 = coco91to80

        ids = list(sorted(self.coco.imgs.keys()))
        if mode == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = FilterCOCOImages(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

        self.transforms_image = transforms_image
        self.transforms_label = transforms_label
        self.transforms_all = transforms_all
        self.make_labels = make_labels

    def ParseTargets(self, img_id: int, coco_targets: list, w: int = None, h: int = None):
        """
        解析目标值
        :param img_id: 图像索引
        :param coco_targets: coco目标值列表
        :param w: 图片宽度
        :param h: 图片高度
        :return: 解析后的目标值列表
        """
        # 只筛选出单个对象，iscrowd字段为1时代表真实框内包含多个同类目标（例如排放在一起的多本书籍），该类目标性质与模型功能有出入，故忽略
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
        boxes, classes = [], []
        for obj in anno:
            if obj["bbox"][2] > 0 and obj["bbox"][3] > 0:
                boxes.append(obj["bbox"])
                classes.append(self.coco91to80[str(obj["category_id"])])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.as_tensor(classes, dtype=torch.float32).unsqueeze(1)

        # 标注信息格式由[xmin, ymin, w, h]转换为对角坐标形式[xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        if (w is not None) and (h is not None):
            boxes[:, [0, 2]] /= w  # 坐标归一化
            boxes[:, [1, 3]] /= h
            boxes[:, [0, 2]].clamp_(min=0, max=1)
            boxes[:, [1, 3]].clamp_(min=0, max=1)

        anchors = torch.cat((boxes, classes), dim=1)

        target = {"anchors": anchors, "image_id": img_id}

        return target

    def __getitem__(self, item):
        # 通过coco API获取标签
        coco = self.coco
        img_id = self.ids[item]  # 获取对应图片的索引
        ann_ids = coco.getAnnIds(imgIds=img_id)  # 通过图片索引获取标签索引
        coco_target = coco.loadAnns(ann_ids)  # 通过标签索引获取标签

        # 读取图片
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')
        w, h = img.size

        # 解析目标值格式
        target = self.ParseTargets(img_id, coco_target, w, h)
        if self.transforms_image is not None:
            img = self.transforms_image(img)
        if self.transforms_all is not None:
            img, target = self.transforms_all(img, target)
        target["anchors"] = target["anchors"][torchvision.ops.remove_small_boxes(target["anchors"], min_size=0.01)]
        if self.make_labels and self.transforms_label is not None:
            target["anchors"] = self.transforms_label(target["anchors"].unsqueeze(0))
        return img, target

    def __len__(self):
        return len(self.ids)

    def GetHW(self, index=None, img_id=None):
        if index is not None:
            img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_matched(batch):
        features, targets = zip(*batch)

        # 图片打包成tensor: [batch_size, channels, h, w]
        features = torch.stack(features)

        # 重新整合目标值格式
        image_ids = [target['image_id'] for target in targets]
        offset_targets = [target['anchors'][0] for target in targets]
        classes_targets = [target['anchors'][1] for target in targets]
        priors_mask = [target['anchors'][2] for target in targets]

        offset_targets = torch.cat(offset_targets, dim=0)
        classes_targets = torch.cat(classes_targets, dim=0)
        priors_mask = torch.cat(priors_mask, dim=0)

        targets = {'anchors': (offset_targets, classes_targets, priors_mask), 'image_ids': image_ids}

        return features, targets

    @staticmethod
    def collate_fn(batch):
        features, targets = zip(*batch)

        # 图片打包成tensor: [batch_size, channels, h, w]
        features = torch.stack(features)

        # 重新整合目标值格式，anchors: [batch_size, num_targets, 5]，image_ids: [num_targets]
        anchors = [target['anchors'] for target in targets]  # 目标锚框放在一起
        image_ids = [target['image_id'] for target in targets]  # 每个图片的id放一起

        # 由于每个图片包含的目标数量不相等，为了组合成一个矩阵，需要将所有图片的目标数量padding成相同的数量（以最多的那个为基准）
        max_object_num = max([anchor.shape[0] for anchor in anchors])
        for i in range(len(anchors)):
            pad = (0, 0, 0, max_object_num - anchors[i].shape[0])  # 行的末尾padding到最大行数
            anchors[i] = f.pad(anchors[i], pad, 'constant', 0)

        targets = {'anchors': torch.stack(anchors), 'image_ids': image_ids}

        # return tuple(zip(*batch))
        return features, targets

    def ShowImage(self, image_id, image_transforms=None, all_transforms=None):
        path = self.coco.loadImgs(image_id)[0]['file_name']
        img = torchvision.io.read_image(os.path.join(self.img_root, path))
        h, w = img.shape[1], img.shape[2]

        classes = json.load(open('coco80_indices.json', 'r'))
        classes = list(classes.values())

        # 解析目标值格式
        ann_ids = self.coco.getAnnIds(imgIds=image_id)  # 通过图片索引获取标签索引
        coco_target = self.coco.loadAnns(ann_ids)  # 通过标签索引获取标签
        target = self.ParseTargets(image_id, coco_target, w, h)
        target['anchors'][:, -1] -= 1
        if image_transforms is not None:
            img = image_transforms(img)
        if all_transforms is not None:
            img, target = all_transforms(img, target)
        img = img.permute(1, 2, 0)
        target = target['anchors']
        target = torch.cat((target, torch.ones((target.shape[0], 1))), dim=1)
        display(img, target, 0.5, classes=classes)


def FilterCOCOImages(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积都非常小的图片
    :param dataset: 数据集
    :param ids: 数据索引
    :return: 保留的图片索引
    """

    def isAllSmall(annotations):
        # 是否所有真实框的宽或高都非常小
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in annotations)

    def isValidAnnotation(annotations):
        # 判断是否是有效标签，只有目标数量不为0，且不是所有目标都很小，才视作有效标签
        if len(annotations) == 0 or isAllSmall(annotations):
            return False
        return True

    valid_ids = []  # 保存标签有效的图片
    # 遍历所有图片索引，找到其对应标签的索引，获得标签，保留符合要求的图片索引
    for img_id in tqdm(ids, desc='filter images'):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        annotations = dataset.loadAnns(ann_ids)

        if isValidAnnotation(annotations):
            valid_ids.append(img_id)

    return valid_ids


class Traffic(Dataset):
    """
    由KITTI数据集提取出的交通数据集，包含Person、Vehicle、Cyclist三种目标
    由于KITTI数据集部分目标出现频次过少，所以本数据集将目标类型概括为以上三类
    """

    def __init__(self, size=(224, 448), num=7481, root_file=Traffic_file):
        self.size = size
        self.features, self.labels, self.classes = LoadCache(os.path.join(root_file, f'{size[0]}x{size[1]}'), num)
        self.mean = [0.4502, 0.4552, 0.3671]
        self.std = [0.2176, 0.2009, 0.2157]
        print('read ' + str(len(self.features)) + ' training examples')

    def __getitem__(self, item):
        return self.features[item].float(), self.labels[item]

    def __len__(self):
        return len(self.features)

    def GetClasses(self):
        return self.classes


class KITTI(Dataset):
    """
    KITTI数据集主要用于自动驾驶领域，包含大量真实道路场景和各类交通目标
    本数据集为KITTI中用于2D目标检测的数据集，包含7481个训练图像和7518个测试图像
    """

    def __init__(self, size=(224, 448), isTraining=True, num=7481):
        self.size = size
        self.classes = []
        if isTraining:
            self.features, self.labels, self.classes = ReadBaseData(KITTI_file, size, num)
        else:
            self.features = ReadTestData(KITTI_file, size)
        self.mean = [0.4502, 0.4552, 0.3671]
        self.std = [0.2176, 0.2009, 0.2157]
        self.all_classes = ['Pedestrian', 'Truck', 'Car', 'Cyclist', 'Misc', 'Van', 'Tram', 'Person_sitting']
        print('read ' + str(len(self.features)) + (' testing', ' training')[isTraining] + ' examples')

    def __getitem__(self, item):
        return self.features[item].float(), self.labels[item]

    def __len__(self):
        return len(self.features)

    def GetClasses(self):
        return self.classes


def ReadBaseData(root_file, size=(384, 384), num=7481):
    """
    读取由图片和对应txt文件组成的基础数据，转化为可以直接用于训练的张量格式，并得到类别标签
    :param root_file: 数据集文件所在目录
    :param size: 模型接收的图片大小
    :param num: 读取数据的数量，默认为None时读取全部
    :return: 训练图片组成的list，对应标签组成的list，字符串形式的目标类别组成的list
    """
    direction = os.path.join(root_file, f'cache{size[0]}x{size[1]}')
    if os.path.exists(direction):
        return LoadCache(direction, num)

    training_images, labels, classes = [], [], []

    # 所有图片都需要改成指定的大小
    resize = torchvision.transforms.Resize(size=size)

    # 训练数据的文件位置
    image_file = 'data_object_image_2'
    training_img_file = os.path.join(root_file, image_file, 'training', 'image_2')

    # 标签的位置
    label_file = os.path.join(root_file, r'data_object_label_2\training\label_2')

    # 读取训练图片和标签
    for img_fname, label_fname in tqdm(list(zip(list(os.walk(training_img_file))[0][2][:num],
                                                list(os.walk(label_file))[0][2][:num])),
                                       desc='读取训练集图片和标签'
                                       ):
        # 读取图片
        image = torchvision.io.read_image(os.path.join(training_img_file, img_fname))  # [channels, h, w]
        h, w = image.shape[1], image.shape[2]  # 获得原图的高度和宽度
        image = resize(image)
        training_images.append(image)

        # 读取标签
        label = []
        with open(os.path.join(label_file, label_fname), 'r') as file:
            for line in file.readlines():
                context = line.split()
                if context[0] == 'DontCare':
                    # DontCare标签意味着该区域在测试时会被忽略，此处不作为物体的标签
                    continue
                try:
                    idx = classes.index(context[0]) + 1  # 目标类别转化为对应的索引
                except:
                    classes.append(context[0])  # 新出现的目标类别则插入到类别列表中
                    idx = len(classes)  # 初始为1，把0留给背景类
                # 获取用于目标检测的标签，依次是xmin, ymin, xmax, ymax, class
                label.append([float(context[4]), float(context[5]), float(context[6]), float(context[7]),
                              float(idx)])
        label = torch.tensor(label)
        label[:, [0, 2]] /= w  # 转化为相对形式
        label[:, [1, 3]] /= h
        labels.append(label)

    # 由于每个图片包含的目标数量不相等，为了组合成一个矩阵，需要将所有图片的目标数量padding成相同的数量（以最多的那个为基准）
    max_object_num = max([label.shape[0] for label in labels])
    for i in range(len(labels)):
        pad = (0, 0, 0, max_object_num - labels[i].shape[0])  # 行的末尾padding到最大行数
        labels[i] = f.pad(labels[i], pad, 'constant', 0)

    PreserveCache(root_file, training_images, labels)

    return training_images, labels, classes


def ReadTestData(root_file, size=(300, 300)):
    """
    读取测试图片，转化为可以直接用于训练的张量格式，并得到类别标签
    :param root_file: 数据集文件所在目录
    :param size: 模型接收的图片大小
    :return:
    """
    testing_images = []

    # 所有图片都需要改成指定的大小
    resize = torchvision.transforms.Resize(size=size)

    # 测试数据的文件位置
    image_file = 'data_object_image_2'
    testing_img_file = os.path.join(root_file, image_file, 'testing', 'image_2')

    # 读取测试图片
    for img_fname in tqdm(list(os.walk(testing_img_file))[0][2], desc='读取测试图片'):
        image = torchvision.io.read_image(os.path.join(testing_img_file, img_fname))  # [channels, h, w]
        image = resize(image)
        testing_images.append(image)

    return testing_images


def PreserveCache(root_file, training_images, labels):
    """
    将resize的图片和清洗后的标签单独保存，方便下次快速读取
    :param root_file: 数据集目录
    :param training_images: 训练图片
    :param labels: 标签
    :return: 无
    """
    h, w = training_images[0].shape[1], training_images[0].shape[2]
    transfer = transforms.ToPILImage()
    direction = os.path.join(root_file, f'cache{h}x{w}')
    image_dir = os.path.join(direction, 'image')
    label_dir = os.path.join(direction, 'label')
    # 创建缓存目录
    if not os.path.exists(direction):
        os.makedirs(direction)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    with open(os.path.join(direction, 'names.txt'), 'w') as names:
        for i in tqdm(range(len(training_images)), desc='preserving cache'):
            file_name = '%06d' % i
            image = transfer(training_images[i].cpu().clone())
            image.record(os.path.join(image_dir, file_name + '.png'))
            with open(os.path.join(label_dir, file_name + '.txt'), 'w') as label_file:
                for anchor in labels[i]:
                    label_file.write(f'{anchor[0]} {anchor[1]} {anchor[2]} {anchor[3]} {anchor[4]}\n')
            names.write(file_name + '\n')


def LoadCache(root_file, num=7481):
    """
    读取由图片和对应txt文件组成的缓存数据，转化为可以直接用于训练的张量格式，并得到类别标签
    :param root_file: 数据集文件所在目录
    :param num: 读取数据的数量，默认为None时读取全部
    :return: 训练图片组成的list，对应标签组成的list，字符串形式的目标类别组成的list
    """
    training_images, labels, classes = [], [], []
    image_dir = os.path.join(root_file, 'image')
    label_dir = os.path.join(root_file, 'label')

    with open(os.path.join(root_file, 'classes.txt'), 'r') as file:
        classes = [cls.strip() for cls in file.readlines()]

    with open(os.path.join(root_file, 'names.txt'), 'r') as file:
        # 读取图片
        names = file.readlines()[:num]

    for name in tqdm(names, desc='loading cache'):
        name = name.strip()
        # 读取图片
        training_images.append(torchvision.io.read_image(os.path.join(image_dir, name + '.png')))  # [channels, h, w]

        # 读取标签
        label = []
        with open(os.path.join(label_dir, name + '.txt'), 'r') as file:
            for line in file.readlines():
                label.append([float(data) for data in line.strip().split()])
        labels.append(torch.tensor(label))

    return training_images, labels, classes


def TransferKITTI(kitti):
    """
    将KITTI数据集精简为Traffic数据集
    :param kitti: kitti数据集
    :return: 无
    """
    root_file = Traffic_file
    training_images, labels = kitti.features, kitti.labels
    h, w = training_images[0].shape[1], training_images[0].shape[2]
    transfer = transforms.ToPILImage()
    direction = os.path.join(root_file, f'{h}x{w}')
    image_dir = os.path.join(direction, 'image')
    label_dir = os.path.join(direction, 'label')
    # 创建缓存目录
    if not os.path.exists(direction):
        os.makedirs(direction)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    with open(os.path.join(direction, 'names.txt'), 'w') as names:
        for i in tqdm(range(len(training_images)), desc='preserving cache'):
            file_name = '%06d' % i
            image = transfer(training_images[i].cpu().clone())
            image.record(os.path.join(image_dir, file_name + '.png'))
            with open(os.path.join(label_dir, file_name + '.txt'), 'w') as label_file:
                for anchor in labels[i]:
                    cls = anchor[4]
                    if cls == 1 or cls == 8:
                        cls = 1
                    elif cls == 2 or cls == 3 or cls == 6 or cls == 7:
                        cls = 2
                    elif cls == 4:
                        cls = 3
                    else:
                        cls = 0
                    label_file.write(f'{anchor[0]} {anchor[1]} {anchor[2]} {anchor[3]} {cls}\n')

            names.write(file_name + '\n')

    with open(os.path.join(direction, 'classes.txt'), 'w') as file:
        file.write('Person\nVehicle\nCyclist')


def SplitTTV(dataset, ratio=(0.9, 0.05, 0.05), seed=42):
    """
    划分训练集(T)、测试集(T)、验证集(V)
    :param dataset: 数据集
    :param ratio: 划分比率，依次为训练集、测试集、验证集
    :param seed: 划分的随机数种子
    :return: 训练集、测试集、验证集
    """
    # 计算三个数据集的分配比率，先划分两个，第一个是训练集
    train_ratio, test_ratio = ratio[0] / sum(ratio), (ratio[1] + ratio[2]) / sum(ratio)
    train_dataset, test_dataset = random_split(dataset, [round(train_ratio * len(dataset)),
                                                         round(test_ratio * len(dataset))],
                                               generator=torch.Generator().manual_seed(seed))

    # 再从第二个数据集中划分两个，分别是测试集和验证集
    test_ratio, eval_ratio = ratio[1] / (ratio[1] + ratio[2]), ratio[2] / (ratio[1] + ratio[2])
    test_dataset, eval_dataset = random_split(test_dataset, [round(test_ratio * len(test_dataset)),
                                                             round(eval_ratio * len(test_dataset))],
                                              generator=torch.Generator().manual_seed(seed))

    return train_dataset, test_dataset, eval_dataset


def GetStandard(dataset):
    """
    计算数据集的均值和方差
    :param dataset: 数据集
    :return: 均值和方差
    """
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    transforms.ToTensor()

    for X, _ in data_loader:
        for d in range(3):
            mean[d] += (X[:, d, :, :] / 255).mean()
            std[d] += (X[:, d, :, :] / 255).std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def Standard(img, mean=[0.4502, 0.4552, 0.3671], std=[0.2176, 0.2009, 0.2157]):
    """
    对图像进行标准化处理
    COCO:
        mean = [0.471, 0.448, 0.408]
        std = [0.234, 0.239, 0.242]
    KITTI:
        mean = [0.4502, 0.4552, 0.3671]
        std = [0.2176, 0.2009, 0.2157]
    :param mean: 均值
    :param std: 方差
    :param img: 输入图像
    :return: 标准化后的图像
    """
    transform = transforms.Normalize(mean=mean, std=std)
    img = transform(img / 255)
    return img


class LetterBoxResize(nn.Module):
    """
    等比例缩放图片，空余部分用0填充，同时调整相应的真实框位置
    """

    def __init__(self, size=(288, 512), padding_ratio=0.5, normal_resize=0.0):
        """
        :param size: 缩放后的图片大小
        :param padding_ratio: 左（或上）端填充黑边的比例，0~1，-1时代表随机选取
        :param normal_resize: 以给定概率不考虑宽高比的拉伸至指定size
        """
        super().__init__()
        self.size = size
        self.padding_ratio = padding_ratio
        self.normal_resize = normal_resize
        self.resize = transforms.Resize(size=size)

    def forward(self, image, targets, image_only=False):
        if random.random() < self.normal_resize:
            return self.resize(image), targets

        # 计算宽和高的缩放比例，选择最小的那个进行等比例缩放，并在另一维度填充至指定size
        h, w = image.shape[-2], image.shape[-1]
        ratio_h = self.size[0] / h
        ratio_w = self.size[1] / w

        if ratio_h < ratio_w:
            ratio = ratio_h
            dim = 2  # 待填充的维度
        else:
            ratio = ratio_w
            dim = 1
        resize = transforms.Resize(size=(round(h * ratio), round(w * ratio)))
        image = resize(image)  # 等比例缩放
        padding_size = self.size[dim - 1] - image.shape[dim]

        if self.padding_ratio == -1:
            # 随机生成数字当做比例
            self.padding_ratio = random.random()

        padding_1 = round(padding_size * self.padding_ratio)
        padding_2 = padding_size - padding_1  # 计算两侧填充的数量
        if dim == 1:
            padding = torch.nn.ZeroPad2d((0, 0, padding_1, padding_2))
        else:
            padding = torch.nn.ZeroPad2d((padding_1, padding_2, 0, 0))
        image = padding(image)  # 用0填充

        if image_only:
            return image

        # 调整真实框坐标
        anchors = targets['anchors']
        if anchors.shape[0] > 0:
            if dim == 1:
                # 高度方向填充
                anchors[:, [1, 3]] = (padding_1 + anchors[:, [1, 3]] * round(h * ratio)) / self.size[0]
            else:
                # 宽度方向填充
                anchors[:, [0, 2]] = (padding_1 + anchors[:, [0, 2]] * round(w * ratio)) / self.size[1]
            anchors[:, :4].clamp_(max=1, min=0)
            targets['anchors'] = anchors
        return image, targets


def CheckPriors(prior_boxes, targets, num_classes, IOU_threshold=0.5):
    """
    检查预设框与真实框的匹配比例
    :param prior_boxes: 预设框[num_priors, 4]
    :param targets: 真实框[真实框数量, 4]
    :param num_classes: 类别数量
    :param IOU_threshold: 匹配的IOU
    :return: 匹配数量, 目标数量
    """
    classes = targets[:, -1] - 1
    targets = targets[:, :-1]
    iou = IOU(BoxCenterToCorner(prior_boxes), targets)
    max_iou, _ = torch.max(iou, dim=0)
    matched_map = max_iou > IOU_threshold
    matched = torch.zeros((num_classes,), device=prior_boxes.device)
    num_targets = torch.zeros((num_classes,), device=prior_boxes.device)
    for i in range(num_classes):
        ids = classes == i
        cls_matched_map = matched_map[ids]
        num_targets[i] += cls_matched_map.shape[0]
        matched[i] += torch.sum(cls_matched_map)

    return matched, num_targets, max_iou.mean()


def k_means(boxes, k, dist=np.median):
    """
    YOLO使用的k-means方法
    请参考: https://github.com/qqwweee/keras-yolo3/blob/master/kmeans.py
    :param boxes: 需要聚类的bboxes
    :param k: 聚类中心数
    :param dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
    """

    def WH_IoU(wh1, wh2):
        wh1 = wh1[:, None]  # [N,1,2]
        wh2 = wh2[None]  # [1,M,2]
        inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
        return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))

    # 在所有的bboxes中随机挑选k个作为簇的中心。
    clusters = boxes[np.random.choice(box_number, k, replace=False)]

    while True:
        # 计算每个bboxes离每个簇的距离 1-IOU(bboxes, anchors)
        distances = 1 - WH_IoU(boxes, clusters)
        # 计算每个bboxes距离最近的簇中心
        current_nearest = np.argmin(distances, axis=1)
        # 每个簇中元素不在发生变化说明以及聚类完毕
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # 根据每个簇中的bboxes重新计算簇中心
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


def ClusterForPriors(cfg, num_clusters):
    """
    聚类产生预设框
    :param cfg: 模型配置字典
    :param num_clusters: 聚类中心数
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h, w = cfg['h'], cfg['w']
    if platform.system() == 'Windows':
        vocRoot = r'D:\DateSet\VOC'
    else:
        vocRoot = r'/root/autodl-tmp/VOC'
    data_transforms = transforms.Compose([transforms.ToTensor()])
    image_transforms = LetterBoxResize(size=(h, w))
    train = VOCDetection(root_file=vocRoot, mode='trainval', dataset='VOC',
                         transforms_image=data_transforms, transforms_all=image_transforms)
    # test物体数量为13652
    voc_iter = DataLoader(train, batch_size=1, collate_fn=train.collate_fn, shuffle=True, num_workers=7)
    targets = torch.tensor([], device=device)
    for X, Y in tqdm(voc_iter):
        target = Y['anchors'][0, :, :-1].to(device)
        target = target[:, 2:] - target[:, :2]
        targets = torch.cat((targets, target), dim=0)
    print('targets:', targets.shape)

    clusters = k_means(targets.cpu().numpy(), num_clusters)
    dic = {'clusters': clusters.tolist()}
    print(clusters)
    clusters = torch.tensor(clusters)
    _, ids = torch.sort(clusters[:, 0])
    clusters = clusters.clone()[ids]
    print(clusters)
    dic['sorted_clusters'] = clusters.tolist()
    p = clusters * torch.tensor([w, h])
    s = p[:, 0] * p[:, 1]
    torch.set_printoptions(sci_mode=False)
    t = torch.cat((clusters, p, s.unsqueeze(1)), dim=1)
    _, ids = torch.sort(s)
    t = t[ids]
    dic['detail'] = t.tolist()
    print(t)
    json_str = json.dumps(dic)
    with open(f'cluster priors for {h}x{w}.json', 'w') as json_file:
        json_file.write(json_str)


def EvalForPriors(cfg):
    """
    验证预设框覆盖率
    :param cfg: 模型配置字典
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if platform.system() == 'Windows':
        vocRoot = r'D:\DateSet\VOC'
    else:
        vocRoot = r'/root/autodl-tmp/VOC'
    # prior_boxes = GenerateBox_new(cfg['feature_maps_h'], cfg['feature_maps_w'],
    #                               cfg['assign_priors'])
    prior_boxes = GenerateBoxHW(cfg['feature_maps_h'], cfg['feature_maps_w'],
                                cfg['scales'], cfg['aspect_ratios'])
    data_transforms = transforms.Compose([transforms.ToTensor()])
    image_transforms = LetterBoxResize(size=(cfg['h'], cfg['w']))
    train = VOCDetection(root_file=vocRoot, mode='test', dataset='VOC',
                         transforms_image=data_transforms, transforms_all=image_transforms)
    # test物体数量为13652
    voc_iter = DataLoader(train, batch_size=8, collate_fn=train.collate_fn, shuffle=False, num_workers=2)

    classes = train.classes[:-1]
    matched = torch.zeros((len(classes),), device=device)
    num_targets = torch.zeros((len(classes),), device=device)
    step = 0
    means = []
    print(prior_boxes.shape)
    for X, Y in tqdm(voc_iter):
        target = Y['anchors'].reshape(Y['anchors'].shape[0] * Y['anchors'].shape[1], Y['anchors'].shape[2])
        is_object = target[:, -1] > 0
        target = target[is_object]
        m, n, mean_iou = CheckPriors(prior_boxes.to(device), target.to(device), len(classes), IOU_threshold=0.5)
        matched += m
        num_targets += n
        means.append(mean_iou)
        step += 1
        if step % 100 == 0:
            dic = dict(list(zip(classes, (matched / num_targets).tolist())))
            dic_matched = dict(list(zip(classes, matched.tolist())))
            dic_targets = dict(list(zip(classes, num_targets.tolist())))
            print('总平均：', torch.sum(matched) / torch.sum(num_targets))
            print('类平均：', sum([value for value in dic.values()]) / len(dic.values()))
            print('平均iou：', sum(means) / len(means))
            print(dic)
            print(dic_matched)
            print(dic_targets)

    dic = dict(list(zip(classes, (matched / num_targets).tolist())))
    dic_matched = dict(list(zip(classes, matched.tolist())))
    dic_targets = dict(list(zip(classes, num_targets.tolist())))
    print('总平均：', torch.sum(matched) / torch.sum(num_targets))
    print('类平均：', sum([value for value in dic.values()]) / len(dic.values()))
    print('平均iou：', sum(means) / len(means))
    print(dic)
    print(dic_matched)
    print(dic_targets)


START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {'person': 1,
                         'car': 2, 'bus': 3, 'bicycle': 4, 'motorbike': 5, 'aeroplane': 6, 'boat': 7, 'train': 8,
                         'chair': 9, 'sofa': 10, 'diningtable': 11, 'tvmonitor': 12, 'bottle': 13, 'pottedplant': 14,
                         'cat': 15, 'dog': 16, 'cow': 17, 'horse': 18, 'sheep': 19, 'bird': 20}
# PRE_DEFINE_CATEGORIES = {'Person': 1, 'Vehicle': 2, 'Bike': 3}
# PRE_DEFINE_CATEGORIES = {"Pedestrian": 1, "Person_sitting": 1, "Cyclist": 1, "Truck": 2, "Car": 2, "Van": 2, "Tram": 2}


def ConvertXMLToJson(xml_list, xml_dir, json_file):
    """
    将VOC数据集的xml标签转换为COCO数据集的json文件
    :param xml_list: XML文件名列表
    :param xml_dir: XML标签目录
    :param json_file: 目标位置
    """
    def get(root, name):
        vars = root.findall(name)
        return vars

    def get_and_check(root, name, length):
        vars = root.findall(name)
        if len(vars) == 0:
            raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
        if length > 0 and len(vars) != length:
            raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
        if length == 1:
            vars = vars[0]
        return vars

    def get_filename_as_int(filename):
        try:
            filename = os.path.splitext(filename)[0]
            return int(filename)
        except:
            raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))

    list_fp = open(xml_list, 'r')
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        print("Processing %s" % (line))
        xml_f = os.path.join(xml_dir, line + '.xml')
        tree = et.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s' % (len(path), line))

        image_id = get_filename_as_int(filename)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                if category not in ['Misc', 'DontCare']:
                    raise ValueError
                continue

            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = float(get_and_check(bndbox, 'xmin', 1).text)
            ymin = float(get_and_check(bndbox, 'ymin', 1).text)
            xmax = float(get_and_check(bndbox, 'xmax', 1).text)
            ymax = float(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


def RandomSplit(trainval_file, train_file, val_file):
    """
    将数据集拆分为训练集和验证集
    :param trainval_file: 数据集全部标签
    :param train_file: 训练标签文件位置
    :param val_file: 验证标签文件位置
    """
    with open(trainval_file, 'r') as file:
        trainval = file.readlines()
        trainval = [line.strip() for line in trainval]
        print(trainval)
        print(len(trainval))

    val = sorted(random.sample(trainval, 1000))
    train = list(filter(lambda x: x not in val, trainval))
    print(val)
    print(len(val))
    print(train)
    print(len(train))
    with open(train_file, 'w') as file:
        file.write('\n'.join(train))
    with open(val_file, 'w') as file:
        file.write('\n'.join(val))


def GenerateSingleXml(name, split_lines, img_size, class_idx, tar_dir):
    """
    转换单个图片的标签
    :param name: 文件名
    :param split_lines: 标签列表
    :param img_size: 图像分辨率
    :param class_idx: 类别标签
    :param tar_dir: 目标文件夹
    """
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("VOC2007")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(name + '.jpg')
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("The VOC2007 Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("PASCAL VOC2007")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(img_size[1]))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(img_size[0]))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(img_size[2]))
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for split_line in split_lines:
        line = split_line.strip().split()
        if line[0] in class_idx:
            object_new = doc.createElement("object")
            annotation.appendChild(object_new)

            name_ = doc.createElement('name')
            object_new.appendChild(name_)
            name_txt = doc.createTextNode(line[0])
            name_.appendChild(name_txt)

            pose = doc.createElement('pose')
            object_new.appendChild(pose)
            pose_txt = doc.createTextNode("Unspecified")
            pose.appendChild(pose_txt)

            truncated = doc.createElement('truncated')
            object_new.appendChild(truncated)
            truncated_txt = doc.createTextNode("0")
            truncated.appendChild(truncated_txt)

            difficult = doc.createElement('difficult')
            object_new.appendChild(difficult)
            difficult_txt = doc.createTextNode("0")
            difficult.appendChild(difficult_txt)
            # threes-1#
            bndbox = doc.createElement('bndbox')
            object_new.appendChild(bndbox)

            xmin = doc.createElement('xmin')
            bndbox.appendChild(xmin)
            xmin_txt = doc.createTextNode(str(float(line[4])))
            xmin.appendChild(xmin_txt)

            ymin = doc.createElement('ymin')
            bndbox.appendChild(ymin)
            ymin_txt = doc.createTextNode(str(float(line[5])))
            ymin.appendChild(ymin_txt)

            xmax = doc.createElement('xmax')
            bndbox.appendChild(xmax)
            xmax_txt = doc.createTextNode(str(float(line[6])))
            xmax.appendChild(xmax_txt)

            ymax = doc.createElement('ymax')
            bndbox.appendChild(ymax)
            ymax_txt = doc.createTextNode(str(float(line[7])))
            ymax.appendChild(ymax_txt)

    with open(tar_dir + name + '.xml', "wb") as file:
        file.write(doc.toprettyxml(indent='\t', encoding='utf-8'))


def GenerateXML(class_idx, img_dir, labels_dir, tar_dir):
    """
    根据TXT格式的标签创建VOC格式的XML标签，可用于KITTI标签转VOC
    :param class_idx: 类别标签列表
    :param img_dir: 图片文件路径
    :param labels_dir: TXT标签文件路径
    :param tar_dir: 转换后的xml文件的路径
    """
    for parent, dirnames, filenames in os.walk(labels_dir):  # 分别得到根目录，子目录和根目录下文件
        for file_name in tqdm(filenames):
            full_path = os.path.join(parent, file_name)  # 获取文件全路径
            f = open(full_path)
            split_lines = f.readlines()
            name = file_name[:-4]  # 后四位是扩展名.txt，只取前面的文件名
            img_path = os.path.join(img_dir, name + '.png')  # 路径需要自行修改
            img_size = cv2.imread(img_path).shape
            GenerateSingleXml(name, split_lines, img_size, class_idx, tar_dir)

