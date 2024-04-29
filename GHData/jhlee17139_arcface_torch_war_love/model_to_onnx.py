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
from torch2onnx import convert_onnx


def main(args):
    # get config
    cfg = get_config(args.config)
    recognition_model = get_face_recognition_model(cfg)
    recognition_model.load_state_dict(torch.load(cfg.train_weight))
    convert_onnx(recognition_model.eval(), cfg.train_weight, os.path.join(cfg.output, "model.onnx"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Face Recognition Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())


