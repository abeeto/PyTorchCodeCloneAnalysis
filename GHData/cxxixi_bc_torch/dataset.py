import os
import random
# import chainer

import utils as U
import opts

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class SoundDataset(Dataset):
    def __init__(self, sounds, labels, opt, train):
        # self.base = chainer.datasets.TupleDataset(sounds, labels)
        self.opt = opt
        self.train = train
        # self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()
        self.data_list = sounds
        self.label_list = labels
        

    def __getitem__(self, index):
        
        data = self.data_list[index]
        label= self.label_list[index]
        data = self.preprocess(data)#.astype(np.float32)
        data = np.expand_dims(data, axis=0)
        if self.train:
            data = np.expand_dims(data, axis=0)
        label = np.asarray(label) 
        # label = np.asarray(label)
#         print(data.shape,label.shape)
        # print(data, label)
        return data, label

    def __len__(self):
        return len(self.data_list)

    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]
#         funcs = []
#         if self.opt.strongAugment:
#             funcs += [U.random_scale(1.25)]

#         funcs += [U.padding(self.opt.inputLength // 2),
#                   U.random_crop(self.opt.inputLength),
#                   U.normalize(32768.0),
#                   ]

#         else:
#             funcs = [U.padding(self.opt.inputLength // 2),
#                      U.normalize(32768.0),
#                      U.multi_crop(self.opt.inputLength, self.opt.nCrops),
#                      ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    # def get_example(self, i):
    #     if self.mix:  # Training phase of BC learning
    #         # Select two training examples
    #         while True:
    #             sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
    #             sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
    #             if label1 != label2:
    #                 break
    #         sound1 = self.preprocess(sound1)
    #         sound2 = self.preprocess(sound2)

    #         # Mix two examples
    #         r = np.array(random.random())
    #         sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
    #         eye = np.eye(self.opt.nClasses)
    #         label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

    #     else:  # Training phase of standard learning or testing phase
    #         sound, label = self.base[i]
    #         sound = self.preprocess(sound).astype(np.float32)
    #         label = np.array(label, dtype=np.int32)

    #     if self.train and self.opt.strongAugment:
    #         sound = U.random_gain(6)(sound).astype(np.float32)

    #     return sound, label


def setup(opt, split):
    # fsd
    if opt.dataset == "fsd":
        dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}_train.npz'.format(opt.fs // 1000)), allow_pickle=True)
    else:
        dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)), allow_pickle=True)
        
        
    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i == split:
            val_sounds.extend(sounds)
            val_labels.extend(labels)
        else:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    # Iterator setup
    train_data = SoundDataset(train_sounds, train_labels, opt, train=True)
    val_data = SoundDataset(val_sounds, val_labels, opt, train=False)


    trainloader = DataLoader(train_data, batch_size=opt.batchSize)
    valloader = DataLoader(val_data, batch_size=opt.batchSize//opt.nCrops)

    return trainloader, valloader

if __name__ == "__main__":

    opt = opts.parse()
    opt.dataset = "esc50"
    opt.netType = "envnet"
    opt.data = "./datasets/"
    for split in opt.splits:
        setup(opt, split)
