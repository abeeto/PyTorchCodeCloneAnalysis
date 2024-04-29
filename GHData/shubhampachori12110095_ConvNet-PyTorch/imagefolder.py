# -*- coding: utf-8 -*-

from torchvision.datasets import ImageFolder

# Extends ImageFolder dataset class to access image path
class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPath, self).__getitem__(index), self.imgs[index] #return image path