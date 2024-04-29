"""
File: base_dataset.py
Author: VDeamoV
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description:
    Use to load data, can be use to create meta-dataset
"""
from __future__ import print_function
import os

import yaml
import torch.utils.data as data
from PIL import Image


class BaseCustomDataset(data.Dataset):
    """
    BaseCustomDataset:
        Provide the fundamental method for other dataset
        Basic function is following below:
            1. Configurae Dataset with yml file
            2. Output Dataset Configure
            3. Basic Analysis the Dataset

    Example yaml:
        ```
        # yaml_file
        info:
            name: 'example_name'
            data_type: ['jpeg', 'JPEG']     # choose which type files you want to load
                                            # case sensitive

        directory:
            root_dir: '~/Desktop/train' # must be absolute path

        preprocess:
            transforms:
                transform_list: ['Scale']
                image_size: 224
            transforms_target:
                transform_list: []

        ```
    """

    def __init__(self, yml_file_path):
        super(BaseCustomDataset, self).__init__()
        try:
            with open(yml_file_path, 'r') as config_file:
                params = yaml.load(config_file)
        except FileNotFoundError as error:
            print(error)
        finally:
            print("===yml load success ===")

        self.params = params
        self.name = params['info']['name']
        self.data_type = params['info']['data_type']

        self.root_dir = params['directory']['root_dir']

        self.transforms = params['preprocess']['transforms']
        self.transforms_target = params['preprocess']['transforms_target']

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def config_output(self):
        """
        Output the configure information
        """
        print("======info=======")
        print("Dataset Name: ", self.name)
        print("Data_type: ", self.data_type)
        print("====directory====")
        print("root_dir: ", self.root_dir)
        print("===preprocess===")
        print("transforms: ", self.transforms)
        print("transforms_target: ", self.transforms_target)

    def pre_analyse(self, a, b, c):
        """
        Use to pre_analyse the dataset

        Output:
            1. How many classes
            2. How many samples each class
        """
        #  TODO:  <07-04-19, VDeamoV> #
        raise NotImplementedError


class CustomImageDataset(BaseCustomDataset):
    """
    CustomImageDataset:
        Load Image in the custom folder
        Basic function is folllowing below:
    """

    def __init__(self, yml_file_path, debug=False):
        """
        inherit from BaseCustomDataset

        Default to have these params:
            self.name
            self.data_type
            self.root_dir
            self.transforms
            self.transforms_target

        PARAMS:

        yml_file_path : your yaml config_path
        debug : defaut=False
                if debug set true, it will print the label and data_path
                in real time
        """
        super(CustomImageDataset, self).__init__(yml_file_path)
        self.output_x = []
        self.output_y = []
        # create data here
        _walk_instance = os.walk(self.root_dir)
        for root, _, files in _walk_instance:
            for item in files:
                if item.endswith(tuple(self.data_type)):
                    if debug:
                        print("image_path:", os.path.join(self.root_dir, root.split('/')[-1], item))
                        print("label:", root.split('/')[-1])
                    raw_image_path = os.path.join(self.root_dir, root.split('/')[-1], item)
                    self.output_x.append(raw_image_path)
                    self.output_y.append(root.split('/')[-1])


    def __getitem__(self, index):
        # output a batch
        # we don't need to concern about the batch size
        raw_image_batch = Image.open(self.output_x[index])
        batch_x = raw_image_batch
        return batch_x, self.output_y[index]

    def __len__(self):
        return len(self.output_y)
