import torch
import torchvision.transforms as transforms
import numpy as np
import numpy.random as random
import data_loader
import data_augmentation as data_aug
from PIL import Image
from rgbd import RGBDFolder
from transform_mutual import TransformMutual


def get_nyud_train_set(image_size=(480, 640), batch_size=16,
                       root_rgb='data/nyud/train/RGB/',
                       root_depth='data/nyud/train/depth_tensor/',
                       data_augmentation=True
                       ):
    if not data_augmentation:
        image_size = (image_size[1], image_size[0])
        transform_rgb = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
            ])

        def transform_depth(depth):
            depth /= 10
            depth = depth.unsqueeze(0)
            return depth
        transform_mutual = None
        loader_rgb = data_loader.pil_loader

    else:
        transform_rgb = data_aug.transform_rgb
        transform_depth = data_aug.transform_depth
        transform_mutual = TransformMutual(
                            img_size=(480, 640),
                            output_size=image_size,
                            rescale_range=(1, 1.5)
                            )
        loader_rgb = data_loader.opencv_loader

    train_set = RGBDFolder(
        root=root_rgb,
        root_depth=root_depth,
        transform_rgb=transform_rgb,
        transform_depth=transform_depth,
        transform_mutual=transform_mutual,
        loader_rgb=loader_rgb
        )
    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
        )
    return train_set_loader


def get_nyud_test_set(image_size=(256, 256),
                      root_rgb='data/nyud/test/RGB/',
                      root_depth='data/nyud/test/depth_tensor/'
                      ):

    if image_size is not None:
        if not isinstance(image_size, tuple):
            print('get_nyud_train_set: invalid size')
            return -1

    transformer = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    test_set = NYUDFolder(
        root=root_rgb,
        root_depth=root_depth,
        transform=transformer
        )
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4
        )
    return test_set_loader
