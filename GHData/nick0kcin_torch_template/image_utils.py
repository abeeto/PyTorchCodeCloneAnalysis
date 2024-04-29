import itertools

import cv2
import numpy as np


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=np.int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def gaussian2d(shape, rect, scale=1, decay=4.):
    '''
    gaussian2d: draws rotated gaussian
    :param shape: size of patch
    :param rect: rotated bounding box
    :param scale: multilplier (default 1)
    :return: gaussian patch with shape = shape
    '''
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    a = rect[1][0]
    b = rect[1][1]
    angle = np.deg2rad(-rect[2])
    sigma = a / decay, b / decay

    x_n = y * np.cos(angle) + x * np.sin(angle)
    y_n = x * np.cos(angle) - y * np.sin(angle)
    h = np.exp(-(x_n ** 2 / sigma[1] ** 2 + y_n ** 2 / sigma[0] ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h /= np.max(h)
    h *= scale
    return h


def draw_gaussian(image, rotated_rect, rect, scale=1, decay=3.):
    data = gaussian2d((rect[3], rect[2]), rotated_rect, scale, decay)
    patch = image[np.clip(rect[1], 0, image.shape[0]):
                  np.clip(rect[1] + rect[3], 0, image.shape[0]),
            np.clip(rect[0], 0, image.shape[1]):
            np.clip(rect[0] + rect[2], 0, image.shape[1])]
    if patch.shape[0] and patch.shape[1] and rotated_rect[1][0] > 0 and rotated_rect[1][1] > 0:
        image[np.clip(rect[1], 0, image.shape[0]):
              np.clip(rect[1] + rect[3], 0, image.shape[0]),
        np.clip(rect[0], 0, image.shape[1]):
        np.clip(rect[0] + rect[2], 0, image.shape[1])] = \
            np.maximum(patch, data[max(0, -rect[1]): np.clip(rect[1] + rect[3], 0, image.shape[0]) - rect[1],
                              max(0, -rect[0]): np.clip(rect[0] + rect[2], 0, image.shape[1]) - rect[0]])
    return image


def get_rotated_rect(points):
    M = cv2.getAffineTransform(np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ]), points)
    sx = np.sqrt((M[:, 0] ** 2).sum())
    theta = np.arctan2(M[1, 0], M[0, 0])
    msy = M[0, 1] * np.cos(theta) + M[1, 1] * np.sin(theta)
    if np.sin(theta).abs() < 1e-05:
        sy = (M[1, 1] - msy * np.sin(theta)) / np.cos(theta)
    else:
        sy = (msy * np.cos(theta) - M[0, 1]) / np.sin(theta)
    m = msy / sy
    assert (np.array([
        [sx * np.cos(theta), sy * m * np.cos(theta) - sy * np.sin(theta)],
        [sx * np.sin(theta), sy * m * np.sin(theta) + sy * np.cos(theta)]
    ]) - M[:, :2]).abs().mean() < 1e-05
    rect = (tuple(M[:.2]), (sx, sy), np.rad2deg(theta))
    return rect


def get_masks_from_polygon(points, shape, down_sample=1):
    if points:
        masks = []
        for i, (point, _) in enumerate(points):
            p = point[:, None, :] / down_sample
            mask = np.zeros(tuple([i // down_sample for i in shape]), dtype=np.int32)
            cv2.fillConvexPoly(mask, np.int0(p), 1)
            if not masks:
                masks.append(mask)
                continue
            for j, m in enumerate(masks):
                if np.sum((m > 0) * mask) == 0:
                    masks[j] += mask * (i + 1)
                    break
            else:
                masks.append(mask * (i + 1))
        return masks
    return [np.zeros(shape, dtype=np.int32)]


def zipped_masks_to_rot_bboxes(masks, n_objects):
    objects = [masks_to_rot_bboxes((masks == i).sum(axis=-1)) for i in range(1, n_objects + 1)]
    return objects


def masks_to_rot_bboxes(mask):
    if mask.any():
        contours, _ = cv2.findContours((255 * mask).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        return rect
    return None


def iou(rect, rect2):
    x_left = max(rect[0], rect2[0])
    y_top = max(rect[1], rect2[1])
    x_right = min(rect[0] + rect[2], rect2[0] + rect2[2])
    y_bottom = min(rect[1] + rect[3], rect2[1] + rect2[3])
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    iou = intersection_area / (
            rect[2] * rect[3] + rect2[3] * rect2[2] - intersection_area)
    return iou


def intersection(rect, rect2):
    x_left = max(rect[0], rect2[0])
    y_top = max(rect[1], rect2[1])
    x_right = min(rect[2], rect2[2])
    y_bottom = min(rect[3], rect2[3])
    if x_right < x_left or y_bottom < y_top:
        intersection_rect = []
    else:
        intersection_rect = [x_left, y_top, x_right, y_bottom]
    return intersection_rect


def area(rect):
    if not rect:
        return 0
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def move(rect, direction, shape):
    assert 0 <= rect[0] - direction[0] <= shape[1] and 0 <= rect[2] - direction[0] <= shape[1] and 0 <= rect[1] - \
           direction[1] <= shape[0] and 0 <= rect[3] - direction[1] <= shape[0]
    return (rect[0] - direction[0], rect[1] - direction[1],
            rect[2] - direction[0], rect[3] - direction[1])


def ssim(img1, img2, r=5):
    assert img1.shape == img2.shape
    values = []
    for i, j in itertools.product(range(1, img1.shape[0] - 2 * r - 2, r), range(1, img1.shape[1] - 2 * r - 2, r)):
        block1 = img1[i:i + 2 * r + 1, j:j + 2 * r + 1]
        if block1.std() > 30:
            block2 = img2[i:i + 2 * r + 1, j:j + 2 * r + 1]
            mean1 = block1.mean()
            mean2 = block2.mean()
            cov = np.mean((block1 - mean1) * (block2 - mean2))
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            value = (2 * mean1 * mean2 + c1) * (2 * cov + c2) / (
                    (mean1 ** 2 + mean2 ** 2 + c1) * (block1.var() + block2.var() + c2))
            values.append(value)
    return np.mean(values) if values else 1.0
