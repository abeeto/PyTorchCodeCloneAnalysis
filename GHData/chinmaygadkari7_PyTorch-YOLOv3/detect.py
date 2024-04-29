import argparse
import time
import logging
from glob import glob
import os

import torch
from torch import optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import cv2

from model import *
import utils
import process
from cfg_parser import parse_configuration



_VALID_EXT = [
    '.jpg',
    '.jpeg',
    '.png'
]

def _validate_image_file(path):
    valid = True
    if not os.path.isfile(path):
        raise RuntimeError("Specified path %r is a directory. Expected image file" % path)

    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext not in _VALID_EXT:
        raise RuntimeError("Specified file %r is not an valid image file." % path )


def detect(path, config_file, weigths):
    config_file = 'cfg/yolov3.cfg'
    configuration = parse_configuration(config_file)
    model = Darknet(configuration)

    model.load_weights('weights/yolov3.weights')
    model = model.eval() # Always put model in eval mode for detection
    logging.debug("Model weights loaded")
    _validate_image_file(path)

    img_data, org_img = process.preprocess_image(path)

    # predict
    with torch.no_grad():
        prediction = model(img_data)

    output = utils.reduce_prediction(prediction, apply_nms=True)

    input_shape = img_data.shape[-2:] # batch, channel, height, width
    original_shape  = org_img.shape[:-1] # height, width, channel
    output = utils.scale_prediction(output, input_shape=input_shape, target_shape=original_shape)
    predicted_image = utils.plot_predictions(org_img, output)
    return predicted_image, output


def main():
    parser = argparse.ArgumentParser(description='YOLOv3 Object Detector')
    parser.add_argument('path', help='Path to image(s)')
    parser.add_argument('--cfg', default='cfg/yolov3.cfg', help='Path to configuration file')
    parser.add_argument('--weights', default='weights/yolov3.weigths', help='Path to weights file')
    parser.add_argument('--output', '-o', help='Path to save detection result')
    args = parser.parse_args()

    path = args.path
    config_file = args.cfg
    if config_file is None:
        config_file = 'cfg/yolov2.cfg'

    weights = args.weights
    if weights is None:
        weights='weights/yolov3.weights'

    if args.output is None:
        out_dir = 'detections'
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.basename(path)
        output_path = os.path.join(out_dir,name)

    image, output = detect(path, config_file, weights)
    plt.imsave(output_path, image)

if __name__ == '__main__':
    main()
