seed = 42  # for reproducibility
import copy
import enum
import random;
from torchio import SubjectsDataset, ScalarImage, LabelMap, Subject
from torchio.transforms import RescaleIntensity, RandomAffine, Compose

random.seed(seed)
import warnings
import tempfile
import subprocess
import multiprocessing
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

torch.manual_seed(seed)

import torchio as tio
from torchio import AFFINE, DATA

import numpy as np
import nibabel as nib
# from unet import UNet
from scipy import stats
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from IPython import display
from tqdm.notebook import tqdm
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print('TorchIO version:', tio.__version__)

# Dataset
dataset_dir_name = 'ixi_tiny'
dataset_dir = Path(dataset_dir_name)
histogram_landmarks_path = 'landmarks.npy'
images_dir = dataset_dir / 'image'
labels_dir = dataset_dir / 'label'
image_paths = sorted(images_dir.glob('*.nii.gz'))
label_paths = sorted(labels_dir.glob('*.nii.gz'))
# assert len(image_paths) == len(label_paths)

subjects = []
for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        mri=tio.ScalarImage(image_path),
        # brain=tio.LabelMap(label_path),
        image_path=image_path,
        label_path=label_path
    )
    subjects.append(subject)
dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')
# for one_subject in dataset:
#     to_ras = tio.ToCanonical()
#     one_subject_ras = to_ras(one_subject)
#     print('Old orientation:', one_subject.mri.orientation)
#     print('New orientation:', one_subject.mri.orientation)
#     image = one_subject_ras.mri
#     image_copy = copy.deepcopy(image)
#
#     random_affine = tio.RandomAffine(scales=(0.95, 1.1), degrees=5, seed=seed)
#     random_elastic = tio.RandomElasticDeformation(max_displacement=(5, 5, 5), seed=seed)
#     blur = tio.RandomBlur(std=(0, 1), seed=seed)
#     add_noise = tio.RandomNoise(std=5, seed=1)
#
#     image_affine = random_affine(image_copy)
#     image_elastic = random_elastic(image_copy)
#     image_blur = blur(image_copy)
#     image_noisy = add_noise(image_copy)
#
#     plt.figure(1)
#     plt.subplot(2, 2, 1)  # 图一包含1行2列子图，当前画在第一行第一列图上
#     plt.imshow(image_affine.data[0, 80, :, :])
#
#     plt.figure(1)
#     plt.subplot(2, 2, 2)  # 当前画在第一行第2列图上
#     plt.imshow(image_elastic.data[0, 80, :, :])
#     plt.figure(1)
#     plt.subplot(2, 2, 3)  # 当前画在第一行第2列图上
#     plt.imshow(image_blur.data[0, 80, :, :])
#     plt.figure(1)
#     plt.subplot(2, 2, 4)  # 当前画在第一行第2列图上
#     plt.imshow(image_noisy.data[0, 80, :, :])
#     plt.show()
#     print(0)
#
# # plt.imshow(one_subject.mri.data[0, 40, :, :])
#
# print(1)

training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.RandomBlur(std=(0, 1), seed=seed, p=0.1),  # blur 50% of times
    tio.RandomNoise(std=5, seed=1, p=0.5),  # Gaussian noise 50% of times
    tio.OneOf({  # either
        tio.RandomAffine(scales=(0.95, 1.05), degrees=5, seed=seed): 0.75,  # random affine
        tio.RandomElasticDeformation(max_displacement=(5, 5, 5), seed=seed): 0.25,  # or random elastic deformation
    }, p=0.8),  # applied to 80% of images
])

for one_subject in dataset:
    image0 = one_subject.mri
    plt.imshow(image0.data[0, int(image0.shape[1]/2), :, :])
    plt.show()
    break
dataset_augmented = SubjectsDataset(subjects, transform=training_transform)
for one_subject in dataset_augmented:
    image = one_subject.mri
    plt.imshow(image.data[0, int(image.shape[1]/2), :, :])
    plt.show()
    pass
