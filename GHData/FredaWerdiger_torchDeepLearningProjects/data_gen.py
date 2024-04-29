import torch
import nibabel as nb
from torch.utils import data
import numpy as np
from tqdm import tqdm
import skimage
'''
From https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
'''


class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = nb.load(str(img_name)).get_fdata(), nb.load(str(tar_name)).get_fdata()
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)
                self.cached_data.append((img, tar))
        else:
            self.pre_transform_data = []
            for img_name, tar_name in zip(self.inputs, self.targets):
                img, tar = nb.load(str(img_name)).get_fdata(), nb.load(str(tar_name)).get_fdata()
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)
                self.pre_transform_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            if self.pre_transform is not None:
                x, y = self.pre_transform_data[index]
            # Select the sample
            else:
                input_ID = self.inputs[index]
                target_ID = self.targets[index]
                # Load input and target
                x, y = nb.load(input_ID).get_fdata(), nb.load(target_ID).get_fdata()

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

