import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# data_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

])

train_dataset = datasets.ImageFolder(root='./data/train',transform=data_transform)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4)


def show_batch_images(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]

    for i in range(4):
        label_ = labels_batch[i].item()
        image_ = np.transpose(images_batch[i], (1, 2, 0))
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(image_)
        ax.set_title(str(label_))
        ax.axis('off')
        plt.pause(0.01)


plt.figure()
for i_batch, sample_batch in enumerate(train_dataloader):
    show_batch_images(sample_batch)

    plt.show()
