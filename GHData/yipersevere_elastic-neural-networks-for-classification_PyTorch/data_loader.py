"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import scipy
from tiny_imagenet_data_loader import tiny_image_data_loader
import os



def get_train_loader(data, data_dir, batch_size, augment, random_seed, target_size,
                           valid_size=0.1, shuffle=True, show_sample=False, num_workers=4, pin_memory=False, debug=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    if target_size == (299,299,3):
        print("=====> resize CIFAR image to 229*229*3")
        target_resize = (299, 299)
    else:
        target_resize = (224, 224)

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            # transforms.Pad(padding=96, padding_mode='reflect'),
            transforms.Resize(target_resize),
            transforms.ToTensor(),
            normalize
        ])
    if data == "CIFAR10" or data == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform
        )
        print("===========================use CIFAR10 dataset===========================")
    elif data == "cifar100" or data == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform
        )
        print("===========================use CIFAR100 dataset===========================")

    elif data == "tiny_imagenet":
        # tut think station path
        # train_data_path = '/media/yi/e7036176-287c-4b18-9609-9811b8e33769/tiny_imagenet/tiny-imagenet-200/train'
        # narvi path
        # train_data_path = '/root/data/tiny-imagenet-200/train'

        # tut thinkstation
        data = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/tiny_imagenet/tiny-imagenet-200"



        # ---------- DATALOADER Setup Phase --------- #

        # 'Create TinyImage Dataset using ImageFolder dataset, perform data augmentation, transform from PIL Image ' \
        #     'to Tensor, normalize and enable shuffling'

        print("\n\n# ---------- DATALOADER Setup Phase --------- #")
        print("Creating Train and Validation Data Loaders")
        # print("Completed......................")

        # def class_extractor(class_list):
        #     """
        #     Create a dictionary of labels from the file words.txt. large_class_dict stores all labels for full ImageNet
        #     dataset. tiny_class_dict consists of only the 200 classes for tiny imagenet dataset.
        #     :param class_list: list of numerical class names like n02124075, n04067472, n04540053, n04099969, etc.
        #     """
        #     filename = os.path.join(args.data, 'words.txt')
        #     fp = open(filename, "r")
        #     data = fp.readlines()

        #     # Create a dictionary with numerical class names as key and corresponding label string as values
        #     large_class_dict = {}
        #     for line in data:
        #         words = line.split("\t")
        #         super_label = words[1].split(",")
        #         large_class_dict[words[0]] = super_label[0].rstrip()  # store only the first string before ',' in dict
        #     fp.close()

        #     # Create a small dictionary with only 200 classes by comparing with each element of the larger dictionary
        #     tiny_class_dict = {}  # smaller dictionary for the classes of tiny imagenet dataset
        #     for small_label in class_list:
        #         for k, v in large_class_dict.items():  # search through the whole dict until found
        #             if small_label == k:
        #                 tiny_class_dict[k] = v
        #                 continue

        #     return tiny_class_dict



        # Batch Sizes for dataloaders
        # train_batch_size = batch_size  # total 500*200 images, 1000 batches of 100 images each

        train_root = os.path.join(data, 'train')  # this is path to training images folder
        

        # The numbers are the mean and std provided in PyTorch documentation to be used for models pretrained on
        # ImageNet data
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Create training dataset after applying data augmentation on images
        train_dataset = datasets.ImageFolder(train_root, transform=train_transform)

        # # Create training dataloader
        # train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
        #                                                         num_workers=5)


        # list of class names, each class name is the name of the parent folder of the images of that class
        # class_names = train_data.classes
        # num_classes = len(class_names)

        # tiny_class = {'n01443537': 'goldfish', 'n01629819': 'European fire salamander', 'n01641577': 'bullfrog', ...}
        # tiny_class = class_extractor(class_names)  # create dict of label string for each of 200 classes

        # return train_data_loader, tiny_class





        # print("===========================successfully load  tiny-imagenet train data===========================")
        
        # return train_loader
    else:
        print("ERROR =============================dataset should be CIFAR10 or CIFAR100")
        NotImplementedError

    # num_train = len(train_dataset)
    # indices = list(range(num_train))
    # split = int(np.floor(valid_size * num_train))

    # if shuffle:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)

    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)
    if debug:
        print("enter debug mode, load subset of train data")
        train_dataset.train_data=train_dataset.train_data[:5000]
        train_dataset.train_labels=train_dataset.train_labels[:5000]


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    # valid_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, sampler=valid_sampler,
    #     num_workers=num_workers, pin_memory=pin_memory,
    # )


    return train_loader


def get_test_loader(data,
                    data_dir,
                    batch_size,
                    target_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    debug=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    if target_size == (299,299,3):
        print("=====> resize CIFAR image to 229*229*3")
        target_resize = (299, 299)
    else:
        target_resize = (224, 224)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize(target_resize),
        transforms.ToTensor(),
        normalize
    ])

    if data == "CIFAR10" or data == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform
        )
        print("test data, CIFAR10")
    elif data == "CIFAR100" or data == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform
        )
        print("test data, CIFAR100")
    elif data == "tiny_imagenet":

        # tut thinkstation
        global data_path
        data_path = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/tiny_imagenet/tiny-imagenet-200"

        def create_val_folder():


            """
            This method is responsible for separating validation images into separate sub folders
            """
            path = os.path.join(data_path, 'val/images')  # path where validation data is present now
            filename = os.path.join(data_path, 'val/val_annotations.txt')  # file where image2class mapping is present
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

        create_val_folder()  # Call method to create validation image folders
        
        # narvi path
        # validation_root = '/root/data/tiny-imagenet-200/train'
        
        # tut think station path
        validation_root = os.path.join(data_path, 'val/images')  # this is path to validation images folder
        
        # Create validation dataset after resizing images
        dataset = datasets.ImageFolder(validation_root, transform=transform)

        # # Create validation dataloader
        # validation_data_loader = torch.utils.data.DataLoader(validation_data,
        #                                                             batch_size=batch_size,
        #                                                             shuffle=False, num_workers=5)
        
        
        print("===========================successfully load  tiny-imagenet test data===========================")
    else:
        print("ERROR =============================dataset should be CIFAR10 or CIFAR100")
        NotImplementedError        

    if debug:
        print("enter debug mode, load subset of test data")
        dataset.test_data=dataset.test_data[:1000]
        dataset.test_labels=dataset.test_labels[:1000]


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )        

    return data_loader