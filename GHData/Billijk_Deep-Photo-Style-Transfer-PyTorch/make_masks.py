"""
Read segmentation results from image and generate mask tensors
"""
from __future__ import print_function
import argparse
import torch
import cv2
import numpy as np

palette = [
    (0, 0, 255),
    (0, 255, 0),
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (255, 255, 0),
    (128, 128, 128),
    (0, 255, 255),
    (255, 0, 255)
]

size = (468, 700)

parser = argparse.ArgumentParser()
parser.add_argument("content_mask", type=str, help="Path of content image.")
parser.add_argument("style_mask", type=str, help="Path of style image.")
parser.add_argument("output", type=str, help="Path of output mask tensor.")
args = parser.parse_args()

mask_img1 = cv2.resize(cv2.imread(args.content_mask), (size[1], size[0]))
mask_img2 = cv2.resize(cv2.imread(args.style_mask), (size[1], size[0]))

# cluster into one of the standard colors
def encode(rgb):
    dists = []
    for i, std in enumerate(palette):
        d = sum([abs(x - y) for x, y in zip(std, rgb)])
        dists.append((d, i))
    return min(dists)[1]

colors = set()
mat1 = np.zeros(size, dtype=np.int32)
mat2 = np.zeros(size, dtype=np.int32)
for i in range(size[0]):
    for j in range(size[1]):
        c1 = encode(mask_img1[i, j])
        c2 = encode(mask_img2[i, j])
        colors.add(c1)
        colors.add(c2)
        mat1[i, j] = c1
        mat2[i, j] = c2
        
print("Number of categories: {}".format(len(colors)))

in_seg, tar_seg = [], []
for c in colors:
    in_seg.append(np.float32(mat1 == c))
    tar_seg.append(np.float32(mat2 == c))

valid_categories = None
torch.save({"in": torch.Tensor(in_seg), "tar": torch.Tensor(tar_seg), "categories": valid_categories}, args.output)
