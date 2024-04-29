import os
import torch
import CustomDataset
import torch.utils.data
def __main__():
    dataset = CustomDataset.M2Det_320Dataset(path_to_dataset=r'D:\datasets\custom_dataset_test',transforms = None,target_transforms = None)
    for i in range(len(dataset)):
        print(dataset[i])
if __name__ == '__main__':
    __main__()

