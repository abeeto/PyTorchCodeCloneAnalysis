################################################################################################################
# this is a modified version of the code from: "https://pythonprogramming.net/convolutional-neural-networks-deep-learning-neural-network-pytorch/"
################################################################################################################

import os
from tqdm import tqdm
import cv2
import numpy as np


class prep_data():
    def __init__(
            self,
            IMG_SIZE=50,
            images_path="images"):

        self.IMG_SIZE = IMG_SIZE
        self.training_data = []
        self.counter_list = []
        labels = dict()
        for i, string in zip(np.arange(len(os.listdir(images_path))),
                             os.listdir(images_path)):
            labels[images_path + '/' + string] = i
        self.LABELS = labels

    def make_training_data(self):
        print('Preparing data for the CNN')
        for label in self.LABELS:
            counter = 0
            for file_name in tqdm(os.listdir(label)):
                # check if .jpg file
                if "jpg" in file_name:
                    try:
                        path = os.path.join(label, file_name)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append(
                            [np.array(img),
                             np.eye(len(self.LABELS))[self.LABELS[label]]]
                        )
                        counter += 1

                    except Exception as e:
                        print(e)
                        pass
            self.counter_list.append(counter)

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        for key, value in self.LABELS.items():
            print(key, value, self.counter_list[value])
