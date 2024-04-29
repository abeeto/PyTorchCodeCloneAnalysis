# %%
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import torchvision
import torchvision.transforms as transforms
import os
import time
from utils_datasets import Datasets, TRANS, LocalDatasets
from multiprocessing import Pool
import shutil

# %%
def process_lbp(fps):
    img = Image.open(fps[0])
    img = ImageOps.exif_transpose(img)
    img2 = TRANS.get_fill_to(img, shape=2000, fill=0)
    img2 = img2.convert('L')
    img_lbp = Image.fromarray(TRANS.get_lbp_merge(
        img2,
        radius=2,
        point_mult=8,
    ))
    img_lbp.save(fps[1])
    return True


# %%
def get_tire_dataset(
            data_path='/host/ubuntu/torch/tire/tire_500',
            batchsize=32,
            shuffle=False,
            num_workers=16,
            train_ratio=0.8,
            image_size=224,
            # center_crop=0,
            zoom_amount=2.0,
            random_crop_amount=1.2,
            random_hflip=True,
            random_vflip=True,
            limit=0,
            lbp={'radius': 2, 'point_mult': 16, 'methods': ['r', 'g', 'b', 'default', 'uniform', 'ror', 'nri_uniform']},
            # random_crop=True,
            color_jitter=True,
            autoaugment_imagenet=True,
            fill=128,
            ):
    # assert args['data_path'] == '/host/ubuntu/torch/tire/tire_500'
    # print('preparing datasets')
    # print('preprocess lbp merge images')
    dp = os.path.abspath(data_path)
    train_augs = []
    test_augs = []
    
    zoom_shape = int(image_size * max(1.0, random_crop_amount, zoom_amount) // 2 * 2)
    pre_random_crop_shape = int(image_size * max(1.0, random_crop_amount) // 2 * 2)
    _fill = (fill, fill, fill)
    train_augs.extend([
        TRANS.fit_to(shape=zoom_shape, fill=_fill),
        transforms.CenterCrop(pre_random_crop_shape),
        transforms.RandomCrop(size=int(image_size), padding=None, fill=_fill),
    ])
    test_augs.extend([
        TRANS.fit_to(shape=int(image_size * max(1.0, random_crop_amount, zoom_amount) // 2 * 2), fill=_fill),
        # transforms.CenterCrop(int(image_size * max(1.0, random_crop_amount) // 2 * 2)),
        transforms.CenterCrop(size=int(image_size)),
    ])
    if random_hflip:
        train_augs.append(transforms.RandomHorizontalFlip())
    if random_vflip:
        train_augs.append(transforms.RandomVerticalFlip())
    
    
    if isinstance(color_jitter, dict):
        train_augs.append(transforms.ColorJitter(**color_jitter))
    elif color_jitter is True:
        train_augs.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0))
    
    if autoaugment_imagenet:
        train_augs.append(transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET))
    
    # lbp augs will include ToTensor and Normalize
    lbp_augs = []
    _method_count = 3
    if isinstance(lbp, dict):
        lbp_augs.append(TRANS.lbp_merge(**lbp))
        if 'methods' in lbp:
            _method_count = len(lbp['methods'])
    lbp_augs.append(transforms.ToTensor())
    lbp_augs.append(transforms.Normalize([0.5 for i in range(_method_count)], [0.25 for i in range(_method_count)]))
    # elif lbp is True:
    #     lbp_augs.append(TRANS.lbp_merge(radius=1, point_mult=8, methods=['l', 'default', 'uniform']))
    
    ds = LocalDatasets(
        dataset='tire',
        path=dp,
        transform_fns=[
            # transforms.Resize(255),
            # transforms.CenterCrop(224),
            # SquarePad(),
            # transforms.Resize(224),
            # TRANS.fit_to(shape=image_size, fill=(128, 128, 128)),
            # transforms.Grayscale(),
            # transforms.Resize(32),
        ],
        transform_fns_train=[
            # transforms.RandomCrop(224, padding=12, fill=(0, 255, 255)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            # # transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=(0, 255, 255)),
            # *[transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET) for i in range(bool(autoaugment_imagenet))],
            # transforms.RandomHorizontalFlip(),
            *train_augs,
            *lbp_augs,
            # TRANS.lbp_merge(radius=1, point_mult=8, methods=['l', 'default', 'uniform']),
            
        ],
        transform_fns_test=[
            *test_augs,
            *lbp_augs,
            # TRANS.lbp_merge(radius=1, point_mult=8, methods=['l', 'default', 'uniform']),
        ],
        transform_fns_post=[],
        num_labels=5,
        
        batchsize=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        train_ratio=train_ratio,
        limit=limit,
    )
    return ds


# %%
def get_tire_dataset_old(
            data_path='/host/ubuntu/torch/tire/tire_500',
            data_path_target=None,
            autoaugment_imagenet=True,
            force_reload=False,
            ):
    raise NotImplementedError('this implementation is no long supported, do not use')
    # assert args['data_path'] == '/host/ubuntu/torch/tire/tire_500'
    print('preparing datasets')
    print('preprocess lbp merge images')
    dp = os.path.abspath(data_path)
    assert os.path.isdir(dp)
    if isinstance(data_path_target, str):
        dp_save = data_path_target
    else:
        dp_save = '{}_lbp'.format(dp)
    _count = 0
    images_fp_pairs = []
    
    if force_reload and os.path.isdir(dp_save):
        shutil.rmtree(dp_save)
    if not os.path.isdir(dp_save):
        classes = [v for v in os.listdir(dp) if os.path.isdir(os.path.join(dp, v))]
        os.makedirs(dp_save)
        for i, _class in enumerate(classes):
            class_dp = os.path.join(dp, _class)
            class_dp_save = os.path.join(dp_save, _class)
            os.makedirs(class_dp_save)
            fns = os.listdir(class_dp)
            for j, fn in enumerate(fns):
                if any([fn.lower().endswith(v) for v in ['.jpg', '.png']]):
                    _count += 1
                    images_fp_pairs.append([
                        os.path.join(class_dp, fn),
                        os.path.join(class_dp_save, os.path.splitext(fn)[0] + '.png'),
                    ])
                _progress = ((j + 1) / len(fns) + i) / len(classes)
                # print('/rprocessed {} images | {:5.1}%'.format(
                #     _count,
                #     _progress * 100,
                # ), end='')
        with Pool(8) as p:
            print(p.map(process_lbp, images_fp_pairs))
    
    ds_local = LocalDatasets(
        dataset='tire',
        path=dp_save,
        batchsize=10,
        transform_fns=[
            # transforms.Resize(255),
            # transforms.CenterCrop(224),
            # TRANS.pad(1000, fill=0),
            # transforms.pad(1000, 0),
            # transforms.CenterCrop(1000),
            # SquarePad(),
            # transforms.Grayscale(),
            transforms.Resize(224),
            # transforms.CenterCrop(1000),
        ],
        transform_fns_train=[
            transforms.RandomCrop(224, padding=12, fill=(0, 255, 255)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            # transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=(0, 255, 255)),
            # transforms.Lambda(TRANS.lbp_transform),
            # transforms.Lambda(TRANS.lbp_transform),
            *[transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET) for i in range(bool(autoaugment_imagenet))],
            
            # TRANS.lbp_merge(radius=1, point_mult=8, methods=['l', 'default', 'uniform']),
            
            transforms.RandomHorizontalFlip(),
        ],
        transform_fns_test=[
            # TRANS.lbp_merge(radius=1, point_mult=8, methods=['l', 'default', 'uniform']),
        ],
        # transform_fns_post=[
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        # ],
        shuffle=False,
        num_workers=16,
        train_ratio=0.8,
        num_labels=5,
        limit_train=20,
        limit_test=10,
    )
    ds = ds_local
    return ds

# %%