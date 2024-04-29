import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from config import Config

def get_dataloader(dataset_mode = 0,isTrain = True):
    """
    Params:
        dataset_mode : if dataset_mode=0 , cifar10 should be used, else dataset_mode = 1, cifar 100 should be used
        isTrain : if we get train dataloader ,isTrain should be True and if we get test dataloader,isTrain should be False
    
    Return: dataloader (train or test)
    """
    if (dataset_mode != 0 and dataset_mode != 1):
        raise ValueError("dataset_mode shoulde be 0 (cifar10) or 1 (cifar100)")
    
    #Get transforms
    if isTrain:
        transform = transforms.Compose([
            transforms.RandomCrop((Config.RESIZED_WIDTH.value,Config.RESIZED_HEIGHT.value),padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(Config.CIFAR10_MEAN.value if dataset_mode==0 else Config.CIFAR100_MEAN.value,Config.CIFAR10_STD.value if dataset_mode==0 else Config.CIFAR100_STD)
        ])
    
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.RESIZED_WIDTH.value,Config.RESIZED_HEIGHT.value)),
            transforms.ToTensor(),
            transforms.Normalize(Config.CIFAR10_MEAN.value if dataset_mode==0 else Config.CIFAR100_MEAN.value,Config.CIFAR10_STD.value if dataset_mode==0 else Config.CIFAR100_STD)
        ])
    
    #Get dataset
    if dataset_mode == 0:
        dataset = datasets.CIFAR10(root="./data",train=isTrain,download=True,transform=transform)
    
    else:
        dataset = datasets.CIFAR100(root="./data",train=isTrain,download=True,transform=transform)
    print(len(dataset))
    #Get dataloader
    dataloader = DataLoader(dataset,batch_size=Config.BATCHSIZE.value,shuffle=True,num_workers=Config.NUM_WORKERS.value)
    
    return dataloader

# if __name__ == '__main__':
#     dataloader = get_dataloader(0,isTrain=False)
#     for (data,target) in (dataloader):
#         print(data.shape,target)
#         break