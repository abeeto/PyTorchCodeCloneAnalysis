import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from PIL import Image


# The only requirement for datasets is that they must implement:
#   - a __len__ function
#   - a __getitem__ function
class MNISTDataset(Dataset):

    def __init__(self, data_dir, resolution, training=False):
        self.resolution = resolution
        self.training = training

        data = os.listdir(data_dir)
        data = [d for d in data if '.jpg' in d]
        data = [{'image': os.path.join(data_dir, d),
                 'label': int(d.split('_')[0])}
                for d in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]['image']
        image = Image.open(image).convert('RGB')
        image = self.transform(image)

        label = self.data[item]['label']
        label = torch.LongTensor([label])

        return image, label

    # Note that every one of the member transforms (e.g. transforms.Resize()) returns an object.
    # You could just keep a reference to this object in your dataset class and avoid recreating it
    # when acquiring each sample, but I prefer the functional style that I use here, since it is more imperative.
    #
    # There is also a functional version of the transforms library (torchvision.transforms.functional) that provides
    # functional calls (e.g. to_tensor(image), but it lacks the Random transforms and so isn't as clean
    # for data augmentation purposes. That being said, you sometimes need to use it. One example is when doing
    # semantic segmentation data augmentation. You want to do the same augmentation for the input image and its
    # output mask, so you need to create your own random values and then use the functional library to do the
    # transformation (e.g. create a random rotation angle and then use the rotate(image, angle) function on both
    # the input and output)
    def transform(self, image):
        if self.training:
            image = transforms.RandomHorizontalFlip()(image)
            image = transforms.RandomRotation(degrees=15)(image)
        image = transforms.Resize(size=self.resolution)(image)  # resolution is either int or HxW
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)
        return image

        # These transformations (especially the resizing) are unecessary for MNIST, but I just
        # wanted to show some of the capabilities
