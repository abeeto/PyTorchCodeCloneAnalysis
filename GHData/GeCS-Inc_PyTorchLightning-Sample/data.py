import cv2
import torch
import torchvision
from PIL import Image


def get_dataset(env, transform):

    assert env.dataset in ["custom", "imagenet"]

    if env.dataset == "custom":
        dataset = torchvision.datasets.ImageFolder(
            env.dataset_root, transform=transform, loader=cv2_image_loader)
        N = len(dataset)
        num_train = int(N * 0.8)
        num_val = N - num_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val])

    return train_dataset, val_dataset


def cv2_image_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) \
        if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img).convert("RGB")
