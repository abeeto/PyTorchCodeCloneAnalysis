import os

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image


def test_PIL_image():
    src_filename = "/tmp/seg_visualization/P2560.png-image.png"
    image: Image.Image = Image.open(src_filename)
    t = TF.to_tensor(image)
    save_image(t, "/tmp/tensor.png")


def test_PIL_resize():
    src_filename = "/tmp/seg_visualization/P2560.png-image.png"
    image: Image.Image = Image.open(src_filename)
    image = image.resize((256, 256), Image.ANTIALIAS)
    t = TF.to_tensor(image)
    save_image(t, "/tmp/tensor.png")


def test_flip():
    src_filename = "/tmp/seg_visualization/P2560.png-image.png"
    image: Image.Image = Image.open(src_filename)
    image = image.resize((256, 256), Image.ANTIALIAS)

    t = TF.to_tensor(image)
    t = TF.hflip(t)
    t = TF.vflip(t)
    save_image(t, "/tmp/tensor.png")


def test_rotate():
    """
    Counter-clockwise rotation in degrees.
    """
    src_filename = "/tmp/seg_visualization/P2560.png-image.png"
    image: Image.Image = Image.open(src_filename)
    image = image.resize((256, 256), Image.ANTIALIAS)

    t = TF.to_tensor(image)
    for degrees in [0, 90, 180, 270]:
        t_r = TF.rotate(t, degrees)
        save_image(t_r, f"/tmp/tensor-r{degrees}.png")


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # test_PIL_image()
    # test_PIL_resize()
    # test_flip()
    test_rotate()
