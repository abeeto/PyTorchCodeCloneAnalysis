# %%
import torch
import numpy as np
from torchvision import transforms
import cv2
import albumentations as A

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


class GaussianBlur(object):
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, image, force_apply=False):
        image = np.array(image)
        prob = np.random.random_sample()
        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), sigma)

        return image


class DataTransform():
    def __init__(self, input_height=150, jitter_strength=1):

        self.jitter_strength = jitter_strength
        self.input_height = input_height

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength, 0.8 * self.jitter_strength, 0.8 * self.jitter_strength,
            0.1 * self.jitter_strength
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        kernel_size = int(0.1 * self.input_height)
        if kernel_size % 2 == 0:
            kernel_size += 1

        data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))
        data_transforms.append(transforms.ToTensor())

        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


# %% ============================== TESTS =====================================
# import glob
# from PIL import Image
# import matplotlib.pyplot as plt
# data = glob.glob('../data_cat_dog/no_label/*.jpg')
# t = DataTransform()


# for i in range(10):
#     img = Image.open(data[i])
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

#     img_1, img_2= t(img)
#     axes[0].imshow(img_1.permute(1, 2, 0))
#     axes[1].imshow(img_2.permute(1, 2, 0))


# # %%

# %%
