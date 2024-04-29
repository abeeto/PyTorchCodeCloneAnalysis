from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import abc


class BaseDataset(Dataset, abc.ABC):
    def __init__(self, path):
        image_path = [line.rstrip('\n') for line in open(path)]
        self.images = []
        self.labels = []
        for img in image_path:
            self.images.append(img.split()[0])
            self.labels.append(np.float32(img.split()[1]))

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.images)


class TrainSet(BaseDataset):
    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert('RGB')
        score = self.labels[item]
        transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        image = transform(image)
        return image, np.array(score, dtype=np.float32)


class ValidationSet(BaseDataset):
    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert('RGB')
        score = self.labels[item]
        image = transforms.ToTensor()(image)
        return image, np.array(score, dtype=np.float32)


class TestSet(BaseDataset):
    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert('RGB')
        image_name = self.images[item]
        score = self.labels[item]
        image = transforms.ToTensor()(image)
        return image, score, image_name
