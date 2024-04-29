import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.face_recognition import get_face_recognition_model
from dataset import get_test_single_dataloader
from lr_scheduler import PolyScheduler
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from scrfd.scrfd_onnx import SCRFD
from tqdm import tqdm
import cv2
from torchvision import transforms
from xml.etree.ElementTree import Element, ElementTree


def main(args):
    # get config
    cfg = get_config(args.config)

    detector_model = SCRFD(model_file=cfg.scrfd_weight)
    detector_model.prepare(-1)

    recognition_model = get_face_recognition_model(cfg)
    recognition_model.load_state_dict(torch.load(cfg.train_weight))
    recognition_model.cuda()
    recognition_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    class_labels = cfg.labels

    img_names = os.listdir(cfg.inference_input)

    for img_name in tqdm(img_names):
        img_path = os.path.join(cfg.inference_input, img_name)
        img = cv2.imread(img_path)
        bboxes = []

        for _ in range(1):
            bboxes, kpss = detector_model.detect(img, 0.5, input_size=(640, 640))
            # bboxes, kpss = detector.detect(img, 0.5)
            # print('all cost:', (tb - ta).total_seconds() * 1000)
        # print(img_path, bboxes.shape)
        if kpss is not None:
            print(kpss.shape)

        crop_img_list = crop_and_resize(img, np.array(bboxes, dtype=np.int32))

        if len(crop_img_list) == 0:
            continue

        # crop_img_list = np.stack(crop_img_list, axis=0)
        tensor_img_list = []

        for crop_img in crop_img_list:
            tensor_img_list.append(transform(crop_img))

        if len(tensor_img_list) == 0:
            continue

        tensor_img_list = torch.stack(tensor_img_list, dim=0)
        tensor_img_list = tensor_img_list.cuda()

        outputs = recognition_model(tensor_img_list)
        _, predicted = torch.max(outputs, 1)

        predicted = predicted.detach().cpu().tolist()
        bboxes = bboxes.astype(int).tolist()
        tmp_bboxes = []

        for bbox in bboxes:
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
                continue

            if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                continue

            tmp_bboxes.append(bbox)

        tree = bbox_to_xml_format(img_path, predicted, tmp_bboxes, class_labels)
        xml_name = os.path.join(cfg.inference_output, '{}.xml'.format(img_path.split('/')[-1].split('.')[0]))

        with open(xml_name, "wb") as file:
            tree.write(file, encoding='utf-8', xml_declaration=True)


def crop_and_resize(img, bbox_list):
    crop_img_list = []

    for bbox in bbox_list:
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
            continue

        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            continue

        crop_img = cv2.resize(crop_img, (112, 112))
        crop_img_list.append(crop_img)

    return crop_img_list


def bbox_to_xml_format(file_name, predicated_list, bboxes, class_labels, width=720, height=480, depth=3):
    root = Element("annotation")

    folder_anno = Element("folder")
    folder_anno.text = "love_war_frames"
    root.append(folder_anno)

    file_anno = Element("filename")
    file_anno.text = file_name.split('/')[-1]
    root.append(file_anno)

    path_anno = Element("path")
    path_anno.text = "G:\itrc_22\love_war_frames\{}".format(file_name)
    root.append(path_anno)

    source_anno = Element("source")
    database_anno = Element("database")
    database_anno.text = "Love_And_War"
    source_anno.append(database_anno)
    root.append(source_anno)

    size_anno = Element("size")
    width_anno = Element("width")
    width_anno.text = str(width)
    size_anno.append(width_anno)

    height_anno = Element("height")
    height_anno.text = str(height)
    size_anno.append(height_anno)

    depth_anno = Element("depth")
    depth_anno.text = str(depth)
    size_anno.append(depth_anno)
    root.append(size_anno)

    segment_anno = Element("segmented")
    segment_anno.text = str(0)
    root.append(segment_anno)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = bbox

        predicated = predicated_list[i]
        category = class_labels[predicated]

        object_anno = Element("object")
        name_anno = Element("name")
        name_anno.text = category
        object_anno.append(name_anno)

        pose_anno = Element("pose")
        pose_anno.text = "Unspecified"
        object_anno.append(pose_anno)

        truncated_anno = Element("truncated")
        truncated_anno.text = str(0)
        object_anno.append(truncated_anno)

        difficult_anno = Element("difficult")
        difficult_anno.text = str(0)
        object_anno.append(difficult_anno)

        bndbox_anno = Element("bndbox")
        xmin_anno = Element("xmin")
        xmin_anno.text = str(x1)
        bndbox_anno.append(xmin_anno)

        ymin_anno = Element("ymin")
        ymin_anno.text = str(y1)
        bndbox_anno.append(ymin_anno)

        xmax_anno = Element("xmax")
        xmax_anno.text = str(x2)
        bndbox_anno.append(xmax_anno)

        ymax_anno = Element("ymax")
        ymax_anno.text = str(y2)
        bndbox_anno.append(ymax_anno)
        object_anno.append(bndbox_anno)
        root.append(object_anno)

    tree = ElementTree(root)
    return tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Recognition Inference in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())

