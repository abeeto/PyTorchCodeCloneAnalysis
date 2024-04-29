import os

import torch
from torchvision import models, datasets, transforms


class ImageDataLoader(object):
    def __init__(
        self,
        batch_size,
        image_folder,
        image_augmentor,
        dataset_split_list=[
            'train',
            'valid',
            'test']):
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.image_augmentor = image_augmentor
        self.dataset_split_list = dataset_split_list
        self.image_transforms = {x: transforms.Compose(
            self.image_augmentor[x]) for x in self.dataset_split_list}

        self.image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(
                    self.image_folder,
                    x),
                self.image_transforms[x])for x in self.dataset_split_list}

    def get_image_dataloader(self):
        self.image_dataloaders = {
            x: torch.utils.data.DataLoader(
                self.image_datasets[x],
                batch_size=self.batch_size,
                shuffle=True) for x in self.dataset_split_list}
        class_names = self.image_datasets['train'].classes
        return self.image_dataloaders, class_names
