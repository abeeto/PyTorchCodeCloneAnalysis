import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# methods for imbalance datasets:
# 1. class weighting
# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,5]))

# 2. oversampling
def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)

    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))

    sample_weights = [0]*len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader


def main():
    loader = get_loader(root_dir='dataset/cats_dogs/cats_dogs_imbalance/cats_dogs_train', batch_size=2)

    for data, labels in loader:
        print(labels)


if __name__=='__main__':
        main()



