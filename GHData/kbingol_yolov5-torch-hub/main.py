import argparse
import glob
import os

import torch


# TODO : Import Yolo Torch & Terminal Arg Parser.

def prediction(model, images, resolution, output):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model)
    model.classes = 0
    predictions = model(images, resolution)
    return predictions.save(output)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Image Folder (*PNG)')
    parser.add_argument('--output', type=str, help='Output Folder')
    parser.add_argument('--model', type=str, help='Model Path')
    parser.add_argument('--resolution', type=int, help='640 or 1280')
    opt = parser.parse_args()
    return opt


def main_prediction():
    opt = parse_opt()
    print(opt)
    print(opt.source)
    print(opt.output)
    print(opt.model)
    print(opt.resolution)
    image_list = list()
    source = glob.glob(opt.source + "*")
    for i in source:
        image_list.append(i)
    print(image_list)
    return prediction(opt.model, image_list, opt.resolution, opt.output)


if __name__ == "__main__":

    main_prediction()
    print("This is the main program.")
