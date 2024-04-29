import cv2
import numpy as np
import glob
import random
import torchvision
import torch

class Cute:

    def __init__(self):
        self.classes = {}
        '''
        classes: {
            classname: [filename] 
        }
        '''

        self.indexes = {}
        self.idx_to_class_name = []
        self.class_name_to_idx = {}

    def load_dir(self, dir_name, class_names=['black', 'white'], filename="*.jpg"):
        i = 0
        self.idx_to_class_name = class_names
        for class_name in class_names:
            self.indexes[class_name] = 0
            self.classes[class_name] = list(glob.glob(dir_name + "/" + class_name + "/" + filename))
            self.class_name_to_idx[class_name] = i
            i += 1
        return self

    def next(self, batch_size=16, height=128, width=128):
        labels = []
        images = []
        for i in range(batch_size):
            item = np.random.randint(len(self.classes))
            class_name = self.idx_to_class_name[item]
            label = item
            image = cv2.imread(self.classes[class_name][self.indexes[class_name]])[:, :, ::-1]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            image = torchvision.transforms.ToTensor()(image).numpy()
            labels.append(label)
            images.append(image)
            self.indexes[class_name] = (self.indexes[class_name] + 1) % len(self.classes[class_name])
            if self.indexes[class_name] == 0:
                random.shuffle(self.classes[class_name])
        # return: (BS, n_classes), (BS, H, W, 3)
        return np.asarray(labels), np.asarray(images)


class Cute_Data( Dataset ):
    def __init__ (self) :
        super(Cute_Data , self).__init__()
        self.len = min([len(image) , len(mask)])
        self.data = np.ones((self.len ,128,128 , 3 ))
        print("Loading Dataset...")
        for i in tqdm(range(self.len)) :
            self.object[i] = cv2.resize(cv2.imread(path + 'image/' + image[i]),  (128,128))
        self.data =  torch.from_numpy(((self.object/(scale/2))-1 ) ).transpose_(3 , 1).double()

    def __getitem__(self , index   ) :
        return self.object[index] ,  self.target[index]

    def __len__(self):
        return self.len

    def load_dir(self, dir_name, class_names=['black', 'white'], filename="*.jpg"):
        path = dir_name
        i = 0
        self.idx_to_class_name = class_names
        for class_name in class_names:
            self.indexes[class_name] = 0
            self.classes[class_name] = list(glob.glob(dir_name + "/" + class_name + "/" + filename))
            self.class_name_to_idx[class_name] = i
            i += 1
        return self
