import torch
import numpy as np
from tqdm import tqdm
import cv2
import csv
import random
from torch.utils.data import Dataset

#############################################################################
# In this file you can specify how many images you want to use 'use_images'
# be carefull not to give a number higher than you have available data in 'data_file'
# 'number_to_split_datasets' serves as a number by which you will split dataset
# to training and validation. f.e if we have dataset of length 1000 and
# number_to_split_dataset is of value 800, then 800 images will be used
# as a training data and 200 will be used a validation data
#############################################################################
# Location of csv file where we have saved out data from simulator
data_file = "driving_log_2_extended_COMBI.csv"
DATA = [0,0,0,0] # Save space for IMAGE, ANGLE, THROTTLE
prepare_batch_X, prepare_batch_y, prepare_batch_z = [],[],[]
# How many images we want to use
use_images = 34340
# How many images will be used for training/validation
number_to_split_datasets = 34100
#############################################################################

# Create dataset class
class Data(Dataset):
    def __init__(self, images, targets, speed):
        self.x = images
        self.y = targets
        self.z = speed
        self.len = len(self.x)
    def __getitem__(self,index):
        return self.x[index], self.y[index], self.z[index]
    def __len__(self):
        return self.len

# Read CSV File and save the data
with open(f'{data_file}', 'r') as file:
    reader = csv.reader(file)
    # Shuffle the data
    shuffler = []
    i = 0
    for row in reader:
        shuffler.append(row)

        if(i > use_images):
            break
        i += 1
    random.shuffle(shuffler)

    # One by one append the shuffled data to our data array
    for row in shuffler:
        if DATA[0] == 0:
            DATA[0],DATA[1],DATA[2], DATA[3]  = [row[0]],[row[3]], [row[4]], [row[6]]
            init = False
            continue
        DATA[0].append(row[0])
        DATA[1].append(row[3])
        DATA[2].append(row[4])
        DATA[3].append(row[6])

# Read all the images and prepare final dataset
for i in tqdm(range(0, len(DATA[0]), 1)):
    # Images convert to grayscale
    path_to_image = DATA[0][i]
    #img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(path_to_image)
    # Prepare images
    prepare_batch_X.append(np.array(img))
    # Prepare steering angle and throttle data
    prepare_batch_y.append(np.array([  float(DATA[1][i]),float(DATA[2][i])  ]))
    # Prepare speed data
    prepare_batch_z.append(np.array([  float(DATA[3][i]) ]))


#Format data for training
# format image data /255.0 so our pixels are in range (0-1)
prepare_batch_X = torch.tensor(prepare_batch_X, dtype=torch.float).view(-1,3,40,80)/255.0
prepare_batch_y = torch.tensor(prepare_batch_y, dtype=torch.float).view(-1, 1, 2)
# format our speed data so it is in range (0-1)
prepare_batch_z = torch.tensor(prepare_batch_z, dtype=torch.float).view(-1, 1)/30.0

print(f"prepare_batch_X.shape: {prepare_batch_X.shape}, prepare_batch_y.shape: {prepare_batch_y.shape}")

# Split data for training/validation
X_train = prepare_batch_X[:number_to_split_datasets]
y_train = prepare_batch_y[:number_to_split_datasets]
z_train = prepare_batch_z[:number_to_split_datasets]
X_test = prepare_batch_X[number_to_split_datasets:]
y_test = prepare_batch_y[number_to_split_datasets:]
z_test = prepare_batch_z[number_to_split_datasets:]

# Create the datasets which will be use for training validation *this data will be exported to train file
dataset_train = Data(X_train, y_train, z_train)
dataset_test = Data(X_test, y_test, z_test)
