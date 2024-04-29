# %%
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# %%
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# %%
#  train & validation dataset
train_loader, val_loader = LoadData(
    dataset=datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
).get_dataloader([0.8, 0.2])

#  test dataset
test_loader = LoadData(
    dataset=datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
).get_dataloader()


# %%
#  Cifar-10's classes
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# %% start from here!
if __name__ == "__main__":
    # setting
    # model = models.resnet50(pretrained=True)
    model = ImageClassificationModel(0, nout=len(classes))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    modelwrapper.train(train_loader, val_loader, max_epochs=5)
    # # resume training
    modelwrapper.train(train_loader, val_loader, max_epochs=20)

    # evaluate the model
    modelwrapper.classification_report(test_loader, classes, visualize=True)
