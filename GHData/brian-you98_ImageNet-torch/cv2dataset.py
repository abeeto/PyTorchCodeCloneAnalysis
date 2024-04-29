import os
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset


# 图片缩放，使用opencv
def letterbox(im, new_shape=(224, 224), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# 使用cv2设置数据集
class CV2Dataset(Dataset):
    def __init__(self, path, img_size):
        self.path = [os.path.join(path, i) for i in os.listdir(path)]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.path[index]
        img_name = os.path.basename(img_path)
        img_label = img_name.split('.')[0]
        label = 0 if img_label == 'cat' else 1
        label = np.array([label], dtype=np.float32)
        img = cv2.imread(img_path)
        img, _, _ = letterbox(img, self.img_size, auto=False)
        data = img.transpose((2, 0, 1))[::-1]
        data = data / 255               # 0-255 -> 0-1
        # data = (data - 0.5) / 0.5     # 0-1 -> -1-1
        data = np.float32(data)
        return data, label

    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    test = CV2Dataset('E:/DataSources/DogsAndCats/train', 224)
    test.__getitem__(40)
