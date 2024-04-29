import os
import io
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.exposure import equalize_hist
from IPython import embed
import skimage.transform

class TensorKartDataSet(Dataset):
    def __init__(self, images, tags, hist_eq=False, diff=False):
        self.images = images
        self.tags = tags
        self.hist_eq = hist_eq
        self.diff = diff
    def __len__(self):
        return len(self.images)


    def __getitem__(self,index):
        image = self.images[index].astype(np.float32) / 255
        if self.diff:
            index = max(1,min(index,self.__len__() - 2))
            last_image = self.images[index-1].astype(np.float32) / 255
            image = image - last_image
        if self.hist_eq:
            image = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
            image = image.copy()
            image = equalize_hist(image)
            image = image.reshape(320,240,1).astype(np.float32)
        tag = self.tags[index].astype(np.float32)
        tag[0] /= 80
        tag[1] /= 80
        
        return {"images" : image,
                "tags" : tag}
