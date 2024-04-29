import os


def pil_loader(path):
    if not os.path.exists(path):
        return None
    from PIL import Image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def opencv_loader(path, gray=False):
    if not os.path.exists(path):
        return None
    import cv2
    import numpy as np
    if not gray:
        img = cv2.imread(path).astype(np.float32)
        img[:, :, [0, 2]] = img[:, :, [2, 0]]
    else:
        img = cv2.imread(path, 0).astype(np.float32)
    return img


def numpy_loader(path):
    if not os.path.exists(path):
        return None
    import numpy as np
    return np.load(path)


def torch_tensor_loader(path):
    if not os.path.exists(path):
        return None
    import torch
    return torch.load(path)
