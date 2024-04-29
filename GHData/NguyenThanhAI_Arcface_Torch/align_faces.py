import os
import argparse
from posixpath import basename

from typing import List
from tqdm import tqdm

from PIL import Image

import numpy as np

from mtcnn import MTCNN

from utils_fn import enumerate_images


def get_args():

    parser = argparse.ArgumentParser()

    #parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\hand_faces")
    #parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\aligned_faces_160")

    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\img_celeba")
    parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\img_aligned_celeba")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    images_dir = args.images_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    images_list: List[str] = enumerate_images(images_dir=images_dir)

    mtcnn = MTCNN()

    for image in tqdm(images_list):
        img = Image.open(image).convert("RGB")
        #bounding_boxes, faces = mtcnn.align_multi(img=img)
        #print(len(bounding_boxes), len(faces))
        face = mtcnn.align(img)
        
        if face is None:
            continue
        '''face = face.resize((160, 160))
        rel_path = os.path.join(*image.split(os.sep)[-2:])
        rel_dir = os.path.dirname(rel_path)
        img_save_dir = os.path.join(save_dir, rel_dir)
        basename = os.path.basename(image)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir, exist_ok=True)
        save_path = os.path.join(img_save_dir, basename)'''
        base_name = os.path.basename(image)
        save_path = os.path.join(save_dir, base_name)
        face.save(save_path)
        