# -*- coding: utf-8 -*-
"""
Created on August 2022

Images reader and data augmentation.

@authors: Nuria Gómez-Vargas
"""


import os
from tkinter import E
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF # https://pytorch.org/vision/main/transforms.html
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import random
import numpy as np
import pickle


##########################################################################################

# SEED

seed = 16
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##########################################################################################

def feature_extract(img):
    """
    It transforms an image to the features extracted via transfer learning.
    
    Parameters
    ----------
    img: np.array
        Matrix of pixels.

    Returns
    -------
    feat: torch.Tensor
        Features extracted by ResNet50.
    """

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model_conv_extraction = resnet50(weights=weights)
    model_conv_extraction.fc = nn.Flatten() #sustituyo la top layer (fc) por lo que yo quiera
    model_conv_extraction.eval()
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    img = preprocess(img)
    feat = model_conv_extraction(img.unsqueeze(0)).squeeze(0) #SAVING FEATURES EXTRACTED OF THE IMAGE

    return feat


def load_dataset(dataset_path, n_test = 3, n_train = 2, augment = 0):
    """
    Loads the dataset and stores the available images for each ray.
    Optionally performs data augmentation.
    Directly saves extracted features of those images.
    The function divides the images of each ray in the train, validation and test sets.
    First, it extracts n_test images of each ray for test.
    The rest of the images are divided in train and validation with a 75% - 25% split.

    Parameters
    ----------
    dataset_path: str
        Self explanatory.
    n_test: int
        Minimum number of images per individual for test.
    n_train: int
        Minimum number of images per individual for training.
    augment: int
        Number of times to perform data augmentation over train set.

    Returns
    -------
    train_dict: dict
        Self explanatory
    valid_dict: dict
        Self explanatory
    test_dict: dict
        Self explanatory
    input_size: int
        Size of the vector of features that ResNet50 returns.
    """

    print("\nWe load the images data set. Will perform data augmentation ", augment, " times.")
    
    #train_dict, valid_dict, test_dict = {}, {}, {}

    for data_source in os.listdir(dataset_path):
        print("Extracting images from the source folder: " + data_source)
        source_path = os.path.join(dataset_path, data_source)
        #for ray in os.listdir(source_path):
        for ray in ['skate_DESTAC-RUN-20-09', 'skate_DESTAC-RUN-20-10', 'skate_DESTAC-RUN-20-12', 'skate_DESTAC-RUN-20-14', 'skate_DESTAC-RUN-20-17', 'skate_DESTAC-RUN-20-19', 'skate_DESTAC-RUN-20-20', 'skate_DESTAC-RUN-20-25', 'skate_DESTAC-RUN-20-27', 'skate_DESTAC-RUN-20-29', 'skate_DESTAC-RUN-20-30', 'skate_DESTAC-RUN-20-34', 'skate_DESTAC-RUN-20-36', 'skate_DESTAC-RUN-20-37', 'skate_DESTAC-RUN-20-38', 'skate_DESTAC-RUN-20-40', 'skate_IGENTAC-RUN-21-01', 'skate_IGENTAC-RUN-21-02', 'skate_IGENTAC-RUN-21-04', 'skate_IGENTAC-RUN-21-05', 'skate_IGENTAC-RUN-21-06', 'skate_IGENTAC-RUN-21-07', 'skate_IGENTAC-RUN-21-10', 'skate_IGENTAC-RUN-21-12', 'skate_IGENTAC-RUN-21-13', 'skate_IGENTAC-RUN-21-14', 'skate_IGENTAC-RUN-21-15', 'skate_IGENTAC-RUN-21-16', 'skate_IGENTAC-RUN-21-17', 'skate_IGENTAC-RUN-21-18', 'skate_IGENTAC-RUN-21-19', 'skate_IGENTAC-RUN-21-20', 'skate_IGENTAC-RUN-21-21', 'skate_IGENTAC-RUN-21-22', 'skate_IGENTAC-RUN-21-23', 'skate_IGENTAC-RUN-21-24', 'skate_IGENTAC-RUN-21-25', 'skate_IGENTAC-RUN-21-27']:
            
            train_dict, valid_dict, test_dict = {}, {}, {}

            if ('skate' in ray) and ('UNK' not in ray):
                print("Extracting images of rays: " + ray)
                ray_path = os.path.join(source_path, ray)

                try:

                    if len(os.listdir(ray_path)) == 1: #if I only have pictures from one day
                        subfolder_path = os.path.join(ray_path, os.listdir(ray_path)[0])
                        image_paths = [os.path.join(subfolder_path, image) for image in os.listdir(subfolder_path) if image.lower().endswith('.jpg') or image.lower().endswith('.png')]
                        if len(image_paths) > n_test + n_train:

                            inds_for_eval = random.sample(range(len(image_paths)), n_test)
                            inds_for_eval.sort(reverse=True)
                            test_dict[ray] = [ feature_extract( read_image(image_paths.pop(ind)) ) for ind in inds_for_eval ]

                            inds_for_train = random.sample(range(len(image_paths)), int(0.75 * len(image_paths)))
                            inds_for_train.sort(reverse=True)
                            train_dict[ray] = []
                            for ind in inds_for_train:
                                img_path = image_paths.pop(ind)
                                train_dict[ray].append( feature_extract( read_image(img_path)) )
                                
                                for _ in range(augment): #DATA AUGMENTATION only in train
                                    for policy in [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]:
                                        train_dict[ray].append( feature_extract( T.AutoAugment(policy)(read_image(img_path)) )) #SAVING FEATURES EXTRACTED OF THE AUGMENTED IMAGE
                                            
                            valid_dict[ray] = [feature_extract( read_image(image_path)) for image_path in image_paths]

                            with open(ray+'.pkl','wb') as f:
                                pickle.dump([train_dict, valid_dict, test_dict], f)

                        else:
                            print('not enough')

                    else: #we have recaptures
                        subfolders = os.listdir(ray_path)
                        test_fold = os.path.join(ray_path, subfolders.pop( np.argmin([len(subfolder) for subfolder in subfolders]) )) 
                        
                        remain = []
                        for subfolder in [os.path.join(ray_path,subf) for subf in subfolders]:
                            for image_path in os.listdir(subfolder):
                                remain.append( os.path.join(subfolder,image_path) )

                        if len(remain) > n_train:
                            test_dict[ray] = [ feature_extract( read_image(os.path.join(test_fold,image_path)) ) for image_path in os.listdir(test_fold) ]

                            inds_for_train = random.sample(range(len(remain)), int(0.75 * len(remain)))
                            inds_for_train.sort(reverse=True)
                            train_dict[ray] = []
                            for ind in inds_for_train:
                                img_path = remain.pop(ind)
                                train_dict[ray].append( feature_extract( read_image(img_path)) )
                                
                                for _ in range(augment): #DATA AUGMENTATION only in train
                                    for policy in [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]:
                                        train_dict[ray].append( feature_extract( T.AutoAugment(policy)(read_image(img_path)) )) #SAVING FEATURES EXTRACTED OF THE AUGMENTED IMAGE
                                            
                            valid_dict[ray] = [feature_extract( read_image(image_path)) for image_path in remain]

                            with open(ray+'.pkl','wb') as f:
                                pickle.dump([train_dict, valid_dict, test_dict], f)

                        else:
                            print('not enough')
                except Exception as e: #for example for .xlsx files
                    None
                    print(e)
                    print('fallo')
                    
                
    input_size = resnet50().fc.in_features #fc is top layer
    return train_dict, valid_dict, test_dict, input_size
