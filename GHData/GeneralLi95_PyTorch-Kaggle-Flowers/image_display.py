

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

def show_image(path):
    img = Image.open(path)
    img_arr = np.array(img)
    plt.figure(figsize=(5,5))
    plt.imshow(np.transpose(img_arr, (0, 1, 2)))
    plt.show()

show_image("data/rose/537207677_f96a0507bb.jpg")

transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

total_dataset = datasets.ImageFolder('data', transform=transform)
dataset_loader = DataLoader(dataset=total_dataset, batch_size=100)
items = iter(dataset_loader)
image, label = items.next()


def show_transformed_image(image):
    np_image = image.numpy()
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.show()


show_transformed_image(make_grid(image))
