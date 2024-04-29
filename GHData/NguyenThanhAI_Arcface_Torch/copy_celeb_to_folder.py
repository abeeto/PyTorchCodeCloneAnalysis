import os

import argparse
import random

import shutil

from itertools import groupby
from tqdm import tqdm


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\img_aligned_celeba")
    parser.add_argument("--anno_file", type=str, default=r"D:\Face_Datasets\identity_CelebA.txt")
    parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\img_aligned_celeba_train_val_1")
    parser.add_argument("--ratio", type=float, default=0.8)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    images_dir: str = args.images_dir
    anno_file: str = args.anno_file
    save_dir: str = args.save_dir
    ratio: float = args.ratio

    random.seed(100)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, "train")):
        os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, "val")):
        os.makedirs(os.path.join(save_dir, "val"), exist_ok=True)

    with open(anno_file, "r") as f:
        annos = f.read()
        lines = annos.split("\n")
        lines = list(filter(None, lines))
        f.close()
    
    images_labels = list(map(lambda x: (x.split(" ")[0], x.split(" ")[1]), lines))
    images_labels.sort(key=lambda x: x[1])
    identity_to_images_list = {}
    

    for identity, images in groupby(images_labels, key=lambda x: x[1]):
        images = list(images)
        identity_to_images_list[identity] = list(map(lambda x: x[0], images))

    #print(identity_to_images_list)

    train_dict = {}
    val_dict = {}

    for identity in identity_to_images_list:
        images = identity_to_images_list[identity]
        random.shuffle(images)
        train_dict[identity] = images[:int(ratio * len(images))]
        val_dict[identity] = images[int(ratio * len(images)):]

    
    for identity in tqdm(identity_to_images_list):
        train_images = train_dict[identity]
        val_images = val_dict[identity]

        train_save_dir = os.path.join(save_dir, "train", str(int(identity) - 1))
        val_save_dir = os.path.join(save_dir, "val", str(int(identity) - 1))

        if not os.path.exists(train_save_dir):
            os.makedirs(train_save_dir, exist_ok=True)

        if not os.path.exists(val_save_dir):
            os.makedirs(val_save_dir, exist_ok=True)

        for image in train_images:
            try:
                shutil.copy2(os.path.join(images_dir, image), os.path.join(train_save_dir, image))
            except Exception as e:
                print("Error: {}".format(e))
                continue

        for image in val_images:
            try:
                shutil.copy2(os.path.join(images_dir, image), os.path.join(val_save_dir, image))
            except Exception as e:
                print("Error: {}".format(e))
                continue
