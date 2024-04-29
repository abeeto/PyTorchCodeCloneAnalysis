import io
import os
from skimage import io
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision

class ImbalancedData(Dataset):
    def __init__(self, csv_file, root_dir, transform= None):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, item):
        img_pth = os.path.join(self.root_dir, self.csv_file.iloc[item, 0])
        image = io.imread(img_pth)
        self.y_label = torch.tensor(int(self.csv_file.iloc[item, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()])


dataset = ImbalancedData(csv_file='imbalanced_dataset_labels.csv', root_dir='imbalanced_dataset',
                             transform=my_transforms)

class_weights = [1,8] #I have 8 cat photos and only 1 dog photo. So I am trying to make balance between classes
# To make class_weights generalizable(because we sometimes might have 100 classes)
# we can use the below code

'''class_weights = []
for labels in self.y_label:
    if len(self.y_label) > 0:
        class_weights.append(1/len(labels))'''

sample_weights = [0] * len(dataset)

for idx, (image,label) in enumerate(dataset):
    class_weight = class_weights[label]
    sample_weights[idx] = class_weight
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

loader = DataLoader(dataset, batch_size=2, sampler=sampler)

for data, labels in loader:
        print(labels)
