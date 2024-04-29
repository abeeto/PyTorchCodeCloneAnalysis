import cv2
import numpy as np
import torch


def rescale_transform(img, factor):
    img = cv2.resize(
        img, (0, 0), img, factor[0], factor[1],
        interpolation=cv2.INTER_LINEAR
        )
    return img


def rotation_transform(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)
    img = cv2.warpAffine(
        img, rotation_matrix, img.shape[1::-1],
        flags=cv2.INTER_LINEAR
        )
    return img


def crop_transform(img, x1, y1, x2, y2):
    return img[x1:x2, y1:y2]


def hflip_transform(img):
    return cv2.flip(img, 1)


def color_transform(img, factor):
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        return img
    img[:, :, 0] *= factor[0]
    img[:, :, 1] *= factor[1]
    img[:, :, 2] *= factor[2]
    return img
