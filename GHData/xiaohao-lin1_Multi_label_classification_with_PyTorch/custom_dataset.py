import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from pathlib import Path
import pandas as pd

inputs_path ={
    'inputs':Path.cwd()/'inputs'/'',
    'img':Path.cwd()/'inputs'/'img',
    'split': Path.cwd()/'inputs'/'split',
    'train_x': Path.cwd()/'inputs'/'split'/'train.txt',
    'val_x': Path.cwd()/'inputs'/'split'/'val.txt',

    'train_y': Path.cwd() / 'inputs' / 'split' / 'train_attr.txt',
    'val_y': Path.cwd() / 'inputs' / 'split' / 'val_attr.txt',

}

outputs_path = {
    'outputs': Path.cwd()/'outputs'
}

#
# with open(inputs_path['train_y'], 'r') as f:
#     print(f.read())
# a = pd.read_csv(inputs_path['train_y'],header=None)
# a.iloc[0]

# print(inputs_path['split']/'val.txt)
#solve the os.path issue
class DeepFashionDataset(Dataset):
    def __init__(self, x_file, y_file, root, transform=None, landmarks=None, bbox=None):
        #annotations is for the y label
        #use numpy to load the fil
        self.x_file = pd.read_csv(x_file)
        self.y_file = pd.read_csv(y_file)
        #root should be /inputs/
        self.root = root
        self.transform = transform
        self.landmarks = landmarks
        self.bbox = bbox

    def __len__(self):
        return len(self.y_file)

    def __getitem__(self, index):
        # TODO: debug the IMG_PATH
        img_path = self.root / str(self.x_file.iloc[index])
        print('img_path', img_path)
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.y_file.iloc[index]))

        if self.transform:
            image = self.transform(image)
        print('image', image)
        print('y_label', y_label)
        return image, y_label

    def load_data(self):


    def read_img_txt_file(self, filepath):
        '''
        :param filepath:
        :return: a list of strings
        '''
        with open(filepath, 'r') as reader:
            lines = reader.read().strip().split()
        #todo: convert to numpy format
        return lines

    def read_digit_lines(self, path):
        '''
            read attr, bbox, and landmarks files
        :param path:
        :return: a list of lists, can
        '''
        #todo: convert to numpy format
        with open(path) as file:
            lines = file.readlines()
            lines = list(filter(lambda x: len(x) > 0, lines))
            lines = list(map(lambda x: (x.strip().split()), lines))
            for i in range(len(lines)):
                lines[i] = list(map(int, lines[i]))
            #lines = np.array(lines)
        return lines

    #def __repr__(self):

#TODO: debug the root
train_data = DeepFashionDataset(
    x_file = inputs_path['train_x'],
    y_file = inputs_path['train_y'],
    root = inputs_path['inputs']
)
print(train_data[0])




