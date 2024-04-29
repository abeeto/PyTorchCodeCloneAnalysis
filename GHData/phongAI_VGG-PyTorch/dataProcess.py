import torch
import os
import torchvision.datasets
from torchvision import io,transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class LoadData():
    @staticmethod
    def LoadDir(path,size = (224,224),batch_size = 8,shuffle = True):
        if os.path.exists(path):
            mean,std = LoadData.calStdMean(path,input_shape=size,batch_size=batch_size)
            #Get transform
            tranform = transforms.Compose([transforms.Resize(size=size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean,std)
                                           ])
            data = torchvision.datasets.ImageFolder(path,transform=tranform)
            dataset = DataLoader(dataset=data,batch_size=batch_size,shuffle=shuffle)
            return dataset
        else:
            raise Exception("Folder is not existed!")

    @staticmethod
    def calStdMean(path,input_shape,batch_size):
        if os.path.exists(path):
            """Load data"""
            dataset = ImageFolder(root=path, transform=transforms.Compose([transforms.Resize(size=input_shape),
                                                                           transforms.ToTensor()
                                                                           ]))
            dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            channels_sum, channel_squared_sum, num_batches = 0, 0, 0

            """Calculate mean and standard value"""
            for data, _ in dataLoader:
                channels_sum += torch.mean(data, dim=[0, 2, 3])
                channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
                num_batches += 1
            mean = channels_sum / num_batches
            std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5

            """Print it out"""
            print(f"Mean value is : {mean.numpy()}")
            print(f"Std value is : {std.numpy()}")
            return mean, std
        else:
            raise Exception("Folder is valid!")



