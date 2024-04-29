
import torch
from torchvision import datasets,transforms
import os



data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalaFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_dir = 'hymenoptera_data'
image_datasets = datasets.ImageFolder(root='/data_dir',transform=data_transforms)
dataloader = torch.utils.data.DataLoader(image_datasets,batch_size=4,shuffle=True,num_workers=4)



data_transforms={
       'train':transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        'val': transforms.Compose([
                transforms.RandomResize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                ]),
    }

data_dir = 'hymenoptera_data'
image_datasets = { x : datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True,num_workers=4) for x in ['train','val']}
datasizes = {x : len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes

