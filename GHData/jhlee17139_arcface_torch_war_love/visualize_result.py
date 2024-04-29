import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.face_recognition import get_face_recognition_model
from utils.utils_config import get_config
from torchvision import transforms
import cv2


def main(args):
    # get config
    cfg = get_config(args.config)

    # labels
    labels = cfg.labels

    # transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    recognition_model = get_face_recognition_model(cfg)
    recognition_model.load_state_dict(torch.load(cfg.train_weight))
    recognition_model.cuda()
    recognition_model.eval()

    inference(cfg, recognition_model, transform, labels)


def inference(cfg, model, transform, labels):
    # read and preprocess the image
    image = cv2.imread(cfg['visualize_input'])
    # get the ground truth class
    gt_class = cfg['visualize_input'].split('/')[-2]
    orig_image = image.copy()
    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.cuda())

    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]
    cv2.putText(orig_image,
                f"GT: {gt_class}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA
                )
    cv2.putText(orig_image,
                f"Pred: {pred_class}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2, cv2.LINE_AA
                )
    print(f"GT: {gt_class}, pred: {pred_class}")
    '''
    cv2.imshow('Result', orig_image)
    cv2.waitKey(0)
    '''
    print('{}'.format(cfg['visualize_output']))
    cv2.imwrite('{}'.format(cfg['visualize_output']), orig_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Face Recognition Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())


