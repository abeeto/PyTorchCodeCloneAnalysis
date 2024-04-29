import os
import glob

import torch

import torch.nn.parallel

import torch.optim

import torch.utils.data
import torch.utils.data.distributed

import cv2
import numpy as np
from pandas import read_csv

import zipfile


def npz_headers(npz):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            yield name[:-4], shape, dtype


class MemapDataset(torch.utils.data.Dataset):

    def __init__(self, path_to_lst_file, data_dir, transform=None):
        """
        path_to_lst_file (string): path to .lst file with format
            img_idx \t img_label \t img_path
        data_dir (string): path to dataset of images, stored as .npy
        transform (callable, optional): transform to be applied on sample
        """
        super(MemapDataset, self).__init__()

        self.lst_file = read_csv(path_to_lst_file, sep="\t")
        self.data_dir = data_dir
        self.transform = transform

        # Retrieve the lazy loader
        self.npz_data_loader = np.load(self.data_dir, mmap_mode="r")
        
    def __len__(self):
        return self.lst_file.shape[0]

    def __getitem__(self, img_idx):
        '''
        Given an image index corresponding to the first column of the lst file, return the image.
        '''
        if torch.is_tensor(img_idx):
            img_idx = img_idx.tolist()

        # Single index
        img_path = str(self.lst_file[self.lst_file["img_idx"] == img_idx]["img_path"].values)
        img_label =  int(self.lst_file[self.lst_file["img_idx"] == img_idx]["img_label"].values)

        # In case of multiple img_idx
        # img_path = list(self.lst_file[self.lst_file["img_idx"].isin(img_idx)]["img_path"].values)
        # img_label = list(self.lst_file[self.lst_file["img_idx"].isin(img_idx)]["img_path"].values)
        
        # Load image
        img_array = self.npz_data_loader(img_path)

        # Apply transforms
        if self.transform:
            img_array = self.transform(img_array)
        
        return img_array, torch.from_numpy(img_label)


class MultiNpzDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, size=(224,224), transform=None):
        """
        data_dir (string): path to dataset of images, stored as .npz
            ex: data-01-1000.npz, data-02-1000.npz
                each file is obtained as follow:
                    np.savez_compressed(
                        "data-01-1000.npz",
                        {
                            "train" : stacked_arrays,
                            "labels": stacked_arrays
                        }
                    )
        size (tuple of int): desired sizes of each image
        transform (callable, optional): transform to be applied on sample
        """
        super(MultiNpzDataset, self).__init__()

        self.data_dir = data_dir
        self.npz_files_list = glob.glob(os.path.join(self.data_dir, "*.npz"))
        self.size = size

        # Retrieve shapes of each file
        self.npz_files_train_shape = []
        self.npz_files_label_shape = []
        for f in self.npz_files_list:
            for key, shape, dty in npz_headers(f):
                if key == "labels":
                    self.npz_files_label_shape.append(shape)
                elif key == "train":
                    self.npz_files_train_shape.append(shape)
                else:
                    raise KeyError("{} key not recognized. Only \"train\" and \"labels\" are valid keys".format(key))

        # Cumsum to help locate specific indices
        self.cum_sum_nb_rows = np.cumsum([s[0] for s in self.npz_files_train_shape])

        # transforms
        self.transform = transform 
        
    def __len__(self):
        return self.cum_sum_nb_rows[-1]

    def __getitem__(self, img_idx):
        '''
        Provided an image index corresponding to the first column of the lst file, return the image.
        '''
        if torch.is_tensor(img_idx):
            img_idx = img_idx.tolist()

        # Retrieve the corresponding .npz file
        file_idx = np.argwhere(self.cum_sum_nb_rows>img_idx).min()

        # Load the corresponding file and extract the row of interest
        file_row = img_idx - self.cum_sum_nb_rows[file_idx-1]

        with np.load(self.npz_files_list[file_idx], allow_pickle=True) as f:
            img_array = f["train"][file_row][:, :, :]
            img_label = f["labels"][file_row]

        # Resize: interpolation param can be
        # INTER_NEAREST: a nearest-neighbor interpolation
        # INTER_LINEAR: a bilinear interpolation (used by default)
        # INTER_AREA: resampling using pixel area relation. It may be a preferred method for image decimation.
        # But when the image is zoomed, it is similar to the INTER_NEAREST method.
        # INTER_CUBIC: a bicubic interpolation over 4x4 pixel neighborhood
        # INTER_LANCZOS4:
        img_array = cv2.resize(img_array, dsize=self.size, interpolation=cv2.INTER_CUBIC)

        # Apply transforms
        if self.transform:
            img_array = self.transform(img_array)
        
        return img_array, int(img_label)