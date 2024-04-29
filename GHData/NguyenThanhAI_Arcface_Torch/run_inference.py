import os
import argparse

from typing import List
from tqdm import tqdm

from PIL import Image

import numpy as np
import cv2

from backbones import get_model
import torch
from torch.utils.data import Dataset, DataLoader

from utils_fn import enumerate_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="r50")
    parser.add_argument("--weights", type=str, default=r"C:\Users\Thanh\Downloads\backbone.pth")
    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\\aligned_faces")
    parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\features")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    network = args.network
    weights = args.weights
    images_dir = args.images_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    images_list: List[str] = enumerate_images(images_dir=images_dir)

    #print(images_list)

    #img = cv2.imread(images_list[0])
    #img = cv2.resize(img, (112, 112))
#
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = Image.open(images_list[0]).convert("RGB").resize((112, 112))
    print(len(images_list))
    img = Image.open(images_list[9]).convert("RGB").resize((112, 112))
    print(images_list[140])
    img = np.array(img)
    #print(img.shape)

    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    model = get_model(name=network)
    #print(model)
    model.load_state_dict(torch.load(weights))
    model.eval()
    model.to(device)

    img = img.to(device=device)
    print(img.shape)
    with torch.no_grad():
        feat_1 = model(img).cpu().numpy()[0]

    feat_1 = feat_1 / np.linalg.norm(feat_1)
    #print(feat_1)
    #print(torch.load(weights).keys())

    #img = cv2.imread(images_list[1])
    img = cv2.imread("output.png")
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    img = img.to(device=device)

    with torch.no_grad():
        feat_2 = model(img).cpu().numpy()[0]
    
    feat_2 = feat_2 / np.linalg.norm(feat_2)

    print(np.clip(np.sum(feat_1 * feat_2), -1, 1))
    #print(np.sqrt(np.sum((feat_1 - feat_2)**2)))