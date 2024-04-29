# -*- coding: utf-8 -*-
"""
@author: roshan
"""

import torchvision.transforms.functional as TF
import cv2
import numpy as np
from PIL import Image

def flip(image, option_value):
    """
    Args:
        image : PIL image
        option_value = random integer between 0 to 3 - flip {0: vertical, 1: horizontal, 2: both, 3: none}
    Return :
        image : PIL image
    """
    if option_value == 0:
        # vertical
        image = TF.vflip(image)
    elif option_value == 1:
        # horizontal
        image = TF.hflip(image)
    elif option_value == 2:
        # horizontally and vertically flip
        image = TF.hflip(image)
        image = TF.vflip(image)
    else:
        image = image
        # no effect
    return image


def rotate(image, angle, option_value, resample_option):
    """
    Args:
        image : PIL image
        angle = any integer value
        option_value = random integer between 0 to 1 - rotate {0: NO rotation, 1: YES rotation}
        resample = PIL.Image.BILINEAR for image and PIL.Image.NEAREST for label
    Return :
        image : PIL image
    """
    if option_value == 1:
        image = TF.rotate(image, angle, interpolation = resample_option)
    else:
        image = image
    return image

def translate(image, x, y, option_value):
    """
    Args:
        image : PIL image
        x and y = translation values along x and y direction
        option_value = random integer between 0 to 1 - rotate {0: NO translation, 1: YES translation}
    Return :
        image : PIL image
    """
    if option_value == 1:
        M = np.float32([[1,0,x],[0,1,y]])
        image = np.asarray(image, dtype = 'int16')
        rows,cols = image.shape[:2]
        image = cv2.warpAffine(image,M,(cols,rows))
        image = Image.fromarray(image.astype(np.uint8))
    else:
        image = image
    return image


def gammaCorrection(image, factor, option_value):
    """
    Args:
        image : PIL image
        factor = any float value
        option_value = random integer between 0 to 1 - {0: NO change, 1: YES change}
        resample = PIL.Image.BILINEAR for image and PIL.Image.NEAREST for label
    Return :
        image : PIL image
    """
    if option_value == 1:
        image = TF.adjust_gamma(image, factor)
    else:
        image = image
    return image