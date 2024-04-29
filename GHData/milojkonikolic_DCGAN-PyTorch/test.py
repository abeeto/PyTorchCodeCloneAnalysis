import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import cv2 as cv

from models.dcgan import Generator
from utils.utils import get_device, get_random_input_vector, postprocess_generated_image


def merge_images(images_list):
    """
    :param images_list: List of images
    :return: merged_images: Merged images. Grid of images
    """
    h, w, c = images_list[0].shape
    num = int(np.sqrt(len(images_list)))
    merged_images = np.zeros((num*h, num*w, c))
    n = 0
    for i in range(num):
        for j in range(num):
            merged_images[i*h:(i+1)*h, j*w:(j+1)*w, :] = images_list[n]
            n += 1
    return merged_images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='',
                        help="Path to generator model")
    parser.add_argument("--in-channels", type=int, default=3,
                        help="Number of channels of the images. Default: 3")
    parser.add_argument("--input-dim", type=int, default=100,
                        help="Input dimension to generator network. Deafult: 100")
    parser.add_argument("--out-dir", type=str, default='',
                        help="Path to output directory where images will be saved")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to generate")
    parser.add_argument("--num-merge", type=int, default=9,
                        help="Merge several generated images into one to make a grid. Default: 9")
    args = parser.parse_args()

    generator = Generator(args.input_dim, args.in_channels)
    generator.load_state_dict(torch.load(args.model))
    generator.eval()
    device = get_device('0')
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    for num in tqdm(range(args.num_images)):
        images_list = []
        for i in range(args.num_merge):
            input_noise = get_random_input_vector(1, args.input_dim, "cpu")
            generated_image = generator(input_noise)
            generated_image = postprocess_generated_image(generated_image)
            images_list.append(generated_image)
        out_img_path = os.path.join(args.out_dir, f"image_{num+1}.jpg")
        if len(images_list) > 1:
            merged_images = merge_images(images_list)
            cv.imwrite(out_img_path, merged_images)
        else:
            cv.imwrite(out_img_path, generated_image)
