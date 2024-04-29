import os
import gc
import numpy as np
import pandas as pd

from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

class test_dataset(Dataset):
    
    def __init__(self, directory, transforms=None, prediction_mode=False):
        # Need some error handling here...
        self.data = self.__draw_directories_map(directory)
        
        # One-hot encoding:
        # if one_hot:
            # Create a placeholder for one-hot encoded labels:
            # labels = np.zeros([self.data.shape[0], self.data['label'].nunique()])
            
            # Create a map to transfer labels:
            # self.one_hot_labels_map = dict()
            # for i, label in enumerate(unique(self.data['label'])):
            #     self.one_hot_labels_map[label] = i
            # one_hot_encoded_labels = pd.get_dummies(self.data['label'])
            # self.data['label'] = one_hot_encoded_labels.values
        
        label_maps = dict()
        for i, label in enumerate(np.sort(self.data['label'].unique())):
            label_maps[label] = i
        
        self.data['label'] = self.data['label'].apply(lambda label: label_maps[label])
        
        # Order by file names for prediction:
        if prediction_mode:
            self.data = self.data.sort_values('data_files', ascending=True)
        
        # Define batch transform functions:
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = torchvision.transforms.transforms.Compose([
                torchvision.transforms.transforms.ToTensor()
            ])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        _row = self.data.iloc[idx]
        # Old:
        _img = Image.open(_row['data_files'])
        # _img = _img.convert("RGB")
        
        # Using cv2:
        # _img = cv2.imread(_row['data_files'])
        # _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        
        # PIL:
        # _img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transforms:
            _img = self.transforms(_img)
        
        _label = _row['label']
        return (_img, _label)
            
    def __draw_directories_map(self, directory):
        contents = pd.DataFrame()
        _sub_folders = os.listdir(directory)
        for i, _folder in enumerate(_sub_folders):
            _tmp_path = os.path.join(directory, _folder)
            for j, (_root_dirs, _child_dirs, _files) in enumerate(os.walk(_tmp_path)):
                _tmp_df = pd.DataFrame({"data_files":[_root_dirs+'/'+_single_file for _single_file in _files], \
                                        "label":_root_dirs.strip("").split("/")[-1]})
                contents = pd.concat([contents, _tmp_df], axis=0)
                del _tmp_df
            gc.collect()

        return contents