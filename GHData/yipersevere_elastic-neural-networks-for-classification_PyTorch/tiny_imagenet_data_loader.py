import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np
import shutil
import os
import argparse
import cv2
from torch.utils.data.sampler import WeightedRandomSampler
# Create an ArgumentParser object which will obtain arguments from command line

# tut thinkstation
# global data


def tiny_image_data_loader(data, args):

    def create_val_folder(data):
        """
        This method is responsible for separating validation images into separate sub folders
        """
        # data = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/tiny_imagenet/tiny-imagenet-200"
        path = os.path.join(data, 'val/images')  # path where validation data is present now
        filename = os.path.join(data, 'val/val_annotations.txt')  # file where image2class mapping is present
        fp = open(filename, "r")  # open file in read mode
        data = fp.readlines()  # read line by line

        # Create a dictionary with image names as key and corresponding classes as values
        val_img_dict = {}
        for line in data:
            words = line.split("\t")
            val_img_dict[words[0]] = words[1]
        fp.close()

        # Create folder if not present, and move image into proper folder
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(path, folder))
            if not os.path.exists(newpath):  # check if folder exists
                os.makedirs(newpath)

            if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
                os.rename(os.path.join(path, img), os.path.join(newpath, img))

    create_val_folder(data)  # Call method to create validation image folders

    # ---------- DATALOADER Setup Phase --------- #

    'Create TinyImage Dataset using ImageFolder dataset, perform data augmentation, transform from PIL Image ' \
        'to Tensor, normalize and enable shuffling'

    print("\n\n# ---------- DATALOADER Setup Phase --------- #")
    print("Creating Train and Validation Data Loaders")
    print("Completed......................")

    def class_extractor(class_list, data):
        """
        Create a dictionary of labels from the file words.txt. large_class_dict stores all labels for full ImageNet
        dataset. tiny_class_dict consists of only the 200 classes for tiny imagenet dataset.
        :param class_list: list of numerical class names like n02124075, n04067472, n04540053, n04099969, etc.
        """
        # data = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/tiny_imagenet/tiny-imagenet-200"
        filename = os.path.join(data, 'words.txt')
        fp = open(filename, "r")
        data = fp.readlines()

        # Create a dictionary with numerical class names as key and corresponding label string as values
        large_class_dict = {}
        for line in data:
            words = line.split("\t")
            super_label = words[1].split(",")
            large_class_dict[words[0]] = super_label[0].rstrip()  # store only the first string before ',' in dict
        fp.close()

        # Create a small dictionary with only 200 classes by comparing with each element of the larger dictionary
        tiny_class_dict = {}  # smaller dictionary for the classes of tiny imagenet dataset
        for small_label in class_list:
            for k, v in large_class_dict.items():  # search through the whole dict until found
                if small_label == k:
                    tiny_class_dict[k] = v
                    continue

        return tiny_class_dict

    # Batch Sizes for dataloaders
    train_batch_size = validation_batch_size = 16  # total 10000 images, 10 batches of 1000 images each

    train_root = os.path.join(data, 'train')  # this is path to training images folder
    validation_root = os.path.join(data, 'val/images')  # this is path to validation images folder

    # The numbers are the mean and std provided in PyTorch documentation to be used for models pretrained on
    # ImageNet data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Create training dataset after applying data augmentation on images
    train_data = datasets.ImageFolder(train_root,
                                        transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                    transforms.ToTensor(),
                                                                    normalize]))
    # Create validation dataset after resizing images
    validation_data = datasets.ImageFolder(validation_root,
                                            transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                            transforms.ToTensor(),
                                                                            normalize]))

    if args.debug:
        # weights for tiny-imagenet, 
        weights = [1] * 200
        # weights = [len(data)for data, label in train_data]
        
        train_sampler = WeightedRandomSampler(weights,num_samples=5000,replacement=True)
        test_sampler = WeightedRandomSampler(weights,num_samples=1000,replacement=True)


        # # see trorch/utils/data/sampler.py
        # class DummySampler(Sampler):
        #     def __init__(self, data):
        #         self.num_samples = len(data)

        #     def __iter__(self):
        #         print ('\tcalling Sampler:__iter__')
        #         return iter(range(self.num_samples))

        #     def __len__(self):
        #         print ('\tcalling Sampler:__len__')
        #         return self.num_samples

        # my_sampler = DummySampler(train_data)

        print("enter debug mode, load subset of train data")
        # train_data.train_data=train_data.train_data[:5000]
        # train_data.train_labels=train_data.train_labels[:5000]

        print("enter debug mode, load subset of test data")
        # validation_data.test_data=validation_data.test_data[:1000]
        # validation_data.test_labels=validation_data.test_labels[:1000]

        # Create training dataloader
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, sampler=train_sampler, shuffle=False,
                                                                num_workers=5)
        # Create validation dataloader
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=validation_batch_size, sampler=test_sampler,
                                                                    shuffle=False, num_workers=5)

    else:


        # Create training dataloader
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                                                num_workers=5)
        # Create validation dataloader
        validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                                    batch_size=validation_batch_size,
                                                                    shuffle=False, num_workers=5)

    # list of class names, each class name is the name of the parent folder of the images of that class
    class_names = train_data.classes
    num_classes = len(class_names)

    # tiny_class = {'n01443537': 'goldfish', 'n01629819': 'European fire salamander', 'n01641577': 'bullfrog', ...}
    # tiny_class = class_extractor(class_names, data)  # create dict of label string for each of 200 classes




    return train_data_loader, validation_data_loader
