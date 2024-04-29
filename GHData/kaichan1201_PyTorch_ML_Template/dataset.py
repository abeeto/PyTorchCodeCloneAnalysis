import os
import sys
import cv2
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset

from albumentations import Compose, Resize, RandomCrop, HorizontalFlip, ShiftScaleRotate, Normalize
from albumentations.pytorch import ToTensor

class CustomData(Dataset):
    def __init__(self, mode, root_dir):
        self.mode = mode
        self.root_dir = root_dir
        self.split = 0.9
        
        # preparing df
        if mode == 'train' or mode == 'val':
            full_df = pd.read_csv(os.path.join(self.root_dir, 'train.csv'))
            # splitting data
            df_groups = full_df.groupby('category')
            # self.data_df = pd.DataFrame(index=full_df.columns.values)
            data_df_list = []
            for g in df_groups.groups.keys():
                group = df_groups.get_group(g)
                if mode == 'train':
                    data_df_list.append(group.head(int(len(group) * self.split)))
                elif mode == 'val':
                    data_df_list.append(group.tail(len(group) - int(len(group) * self.split)))
            self.data_df = pd.concat(data_df_list)

        elif mode == 'test':
            self.data_df = pd.read_csv(os.join(self.root_dir, 'test.csv'))
        else:
            raise NotImplementedError
        self.data_df = self.data_df.reset_index()

        # preparing transform
        if mode == 'train':
            self.transform = Compose([
                Resize(256, 256),
                RandomCrop(224, 224),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(p=0.7),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensor()
            ])
        else:
            self.transform = Compose([
                Resize(224, 224),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensor()
            ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        if self.mode in ['train', 'val']:
            img = cv2.imread(os.path.join(self.root_dir, 'train', 'train',
                                          '{:02d}'.format(row['category']), row['filename']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(image=img)['image']

            gt = torch.tensor(row['category']).long()
            return img, gt
        else:
            img = cv2.imread(os.path.join(self.root_dir, 'test', 'test', row['filename']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(image=img)['image']
            return img, row['filename']


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    df_groups = df.groupby('category')
    print(df_groups.size())
    # for g in df_groups.groups.keys():
    #     group = df_groups.get_group(g)
    #     print(group)

