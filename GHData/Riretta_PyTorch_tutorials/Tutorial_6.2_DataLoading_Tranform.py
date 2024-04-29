#If the images are organized in forlders
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets, utils


data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                     ])

hymenoptera_dataset = datasets.ImageFolder(root='data/hymenoptera_data/train',
                                           transform=data_transform)

dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size = 4,
                                             shuffle=True,
                                             num_workers = 4)
print(len(hymenoptera_dataset))

