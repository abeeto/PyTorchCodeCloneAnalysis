import argparse

import torch
import torchvision
from PIL import Image

import utils
from model import Darknet

to_image = torchvision.transforms.ToTensor()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv4 inference")

    parser.add_argument("--img-file", metavar="origin-img", default="./image/demo.png",
                        help="the image to predict (default: %(default)s)")

    parser.add_argument("--weight", required=True, metavar="/path/to/yolov4.weights", help="the path of weight file")

    parser.add_argument("--save-img", metavar="predicted-img", help="the path to save predicted image")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    img: Image.Image = Image.open(args.img_file)
    img = img.resize((608, 608))

    # C*H*W
    img_data = to_image(img)

    net = Darknet(img_data.size(0))
    net.load_weights(args.weight)
    net.eval()

    with torch.no_grad():
        boxes, confs = net(img_data.unsqueeze(0))

        idxes_pred, boxes_pred, probs_pred = utils.post_processing(boxes, confs, 0.4, 0.6)

    utils.plot_box(boxes_pred, args.img_file, args.save_img)
