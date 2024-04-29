import os
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image

def bounding_box(image, center, patch_h=16, patch_w=16, patch_h_shift=8, patch_w_shift=8):
    """
        Given a numpy image and bounding box center, create random crop while checking for
        edge scenarios.

        image: np_array
        center: tuple (y,x)
        patch_h: int
        patch_w: int
        patch_h_shift: int
        patch_w_shift: int

        return: array[int]
    """


    # store og image info
    height = image.shape[0] - 1
    width = image.shape[1] - 1

    # make sure crop is smaller than image
    assert (patch_h_shift + patch_h) < height
    assert (patch_w_shift + patch_w) < width

    # random shift
    h_shift = random.randrange(-patch_h_shift, patch_h_shift+1)
    w_shift = random.randrange(-patch_w_shift, patch_w_shift+1)

    # shift center point based on random shift values
    center = (max(0, center[0]+h_shift), max(0, center[1]+w_shift))

    # check if left/top edge case
    x1 = max(0, center[1]-patch_w/2)
    y1 = max(0, center[0]-patch_h/2)

    # check right/bottom edge case and alter if above edge case was hit
    x2 = min((center[1]+patch_w/2) if x1 != 0 else width, width)
    y2 = min((center[0]+patch_h/2) if y1 != 0 else height, height)

    # if bottom/right edge case, then we need to change (y1, x1) 
    # to right edge case minus width
    x1 = width-(patch_w) if x2 == width else x1
    y1 = height-(patch_h) if y2 == height else y1

    # return int for slicing
    bbox = [int(x1), int(y1), int(x2), int(y2)]

    return bbox

