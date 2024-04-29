# These are the libraries will be used for this lab.
import torch
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib.pyplot import imshow
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import sys
torch.manual_seed(0)


# Create your own dataset object
class Dataset(Dataset):
    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        # Image directory
        self.data_dir = data_dir

        # The transform is going to be used on image
        self.transform = transform
        data_dircsv_file = os.path.join(self.data_dir, csv_file)
        # Load the CSV file contains image info
        self.data_name = pd.read_csv(data_dircsv_file)

        # Number of images in dataset
        self.len = self.data_name.shape[0]

    # Get the length
    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):
        # Image file path
        img_name = os.path.join(self.data_dir, self.data_name.iloc[idx, 1])
        # Open image file
        image = Image.open(img_name)

        # The class label for the image
        y = self.data_name.iloc[idx, 0]

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y


def show_data(data_sample, shape=(28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))
    plt.show()


# Read CSV file from the URL and print out the first five samples
directory = "/Users/Shiyi/PycharmProjects/Pytorch_course"
csv_file = 'index.csv'
csv_path = os.path.join(directory, csv_file)
data_name = pd.read_csv(csv_path)
print(data_name.head())
# sys.exit(0)

# Get the value on location row 0, column 1 (Notice that index starts at 0)
print('File name:', data_name.iloc[0, 1])
# Get the value on location row 0, column 0 (Notice that index starts at 0.)
print('y:', data_name.iloc[0, 0])
# The number of samples corresponds to the number of rows in a dataframe.
# Print out the total number of rows in training dataset
print('The number of rows: ', data_name.shape[0])  # [0]: rows; [1]: columns
# sys.exit(0)

# Load second image
image_name = data_name.iloc[1, 1]
image_path = os.path.join(directory, image_name)
# # Print second image
# image = Image.open(image_path)
# plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# plt.title(data_name.iloc[1, 0])
# plt.show()

# Create the dataset objects
dataset = Dataset(csv_file=csv_file, data_dir=directory)
print(len(dataset), dataset.data_dir)
image, y = dataset[0]  # image = dataset[0][0]
# plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# plt.title(y)
# plt.show()

# Combine two transforms: crop and convert to tensor. Apply the compose to MNIST dataset
croptensor_data_transform = transforms.Compose(
    [transforms.CenterCrop(20), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file, data_dir=directory,
                  transform=croptensor_data_transform)
print("The shape of the first element tensor: ", dataset[0][0].shape)
show_data(dataset[0], shape=(20, 20))

# Combine two transforms: Vertical flip and convert to tensor.
fliptensor_data_transform = transforms.Compose(
    [transforms.RandomVerticalFlip(p=1), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file, data_dir=directory,
                  transform=fliptensor_data_transform)
show_data(dataset[1])

# Practice: Combine vertical flip, horizontal flip and convert to tensor
# as a compose. Apply the compose on image. Then plot the image
my_data_transform = transforms.Compose(
    [transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1),
     transforms.ToTensor()])
dataset_tf = Dataset(csv_file=csv_file, data_dir=directory,
                     transform=my_data_transform)
show_data(dataset_tf[0])

# Import a prebuilt dataset
dataset = dsets.MNIST(
    root='./data',
    train=False,  # training: True: for training,; False: for testing
    download=True,
    transform=transforms.ToTensor()
)
print(type(dataset[0]))  # <class 'tuple'>
print(len(dataset[0]))  # 2
print(dataset[0][0].shape)  # torch.Size([1, 28, 28])
print(type(dataset[0][0]))  # torch.Tensor
print(dataset[0][1])  # 7, hand written 7
print(type(dataset[0][1]))  # int
show_data(dataset[0])

# Combine vertical flip, horizontal flip and convert to tensor as a compose.
my_data_transform = transforms.Compose(
    [transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1),
     transforms.ToTensor()])
dataset = dsets.MNIST(root='./data', train=False, download=True,
                      transform=fliptensor_data_transform)
show_data(dataset[0])
print(dataset[0][0])  # the first element is a 28 * 28 tensor
print(dataset[0][1])  # the second element is a long tensor that image belongs to
