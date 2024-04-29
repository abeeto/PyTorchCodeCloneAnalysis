import os
import cv2
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

THRESHOLD = 0.9


def color_to_line(img):
    neighbor_hood_8 = np.array([[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1]],
                               np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_dilate = cv2.dilate(img, neighbor_hood_8, iterations=1)
    img = cv2.absdiff(img, img_dilate)
    img = cv2.bitwise_not(img)
    return Image.fromarray(img)


class TwoImgRandomCrop(T.RandomCrop):

    def __call__(self, img1, img2):
        i, j, h, w = self.get_params(img1, self.size)
        return F.crop(img1, i, j, h, w), F.crop(img2, i, j, h, w)


class LineColorDataset(Dataset):

    def __init__(self, color_path, resize=True, size=(512, 512)):
        self.color_path = color_path
        self.colors = os.listdir(color_path)
        self.random_crop = TwoImgRandomCrop(size) if resize else None
        self.normal_crop = T.Resize(size)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        color_img = Image.open(os.path.join(self.color_path,
                                            self.colors[index])).convert('RGB')
        line_img = color_to_line(np.asarray(color_img))
        if self.random_crop is not None:
            if random.random() > THRESHOLD:
                line_img = self.normal_crop(line_img)
                color_img = self.normal_crop(color_img)
            else:
                line_img, color_img = self.random_crop(line_img, color_img)
        return self.transforms(line_img), self.transforms(color_img)

    def __len__(self):
        return len(self.colors)


def test(line_path, color_path, size=(512, 512)):
    import matplotlib.pyplot as plt
    import numpy as np
    resize = TwoImgRandomCrop(size)
    line = Image.open(line_path).convert('L')
    color = Image.open(color_path).convert('RGB')
    line, color = resize(line, color)
    plt.figure()
    plt.imshow(np.asarray(line))
    plt.show()
    plt.figure()
    plt.imshow(np.asarray(color))
    plt.show()
