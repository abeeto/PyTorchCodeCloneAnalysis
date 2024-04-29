from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
# Image transformations
from utils.ImageUtil import showImage, loadImage

__transforms__ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


# 数据增强代码，暂放
def data_enforce(image, box):
    return image, box


class ImageDataSet(Dataset):
    def __init__(self, transform=__transforms__, file_name="./train.txt"):
        file = open(file_name, "r")
        s = file.read().strip()
        file.close()
        self.labels = s.split("\n")
        self.transforms = self.transform = transforms.Compose(transform)

    def __getitem__(self, item):
        line = self.labels[item]
        line = line.split(" ")
        image = loadImage(line[0])
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        image, box = data_enforce(image, box)
        image = self.transforms(image)
        return image, box

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = ImageDataSet()
    _image, _object = dataset[100]
    showImage(_image.numpy())
    print(_object)
