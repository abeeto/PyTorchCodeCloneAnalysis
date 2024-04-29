from PIL import Image
import random

from torch.utils import data
from torchvision.transforms import Compose, Resize, ToTensor


class CSVSet(data.Dataset):
    def __init__(self, csv_path, delim=","):
        self.items = []
        self.preprocess = Compose([
            Resize((256, 256)),
            ToTensor()
        ])

        with open(csv_path, "r") as f:
            lines = f.readlines()

        print("Path:", csv_path)

        for line in lines:
            line = line.strip().split(",")
            self.items += line

    def __getitem__(self, idx):
        img = Image.open(self.items[idx])
        img = self.preprocess(img)

        # TODO : normalization into preprocess
        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1
        return img

    def __len__(self):
        return len(self.items)


def Loader(csv_path, batch_size, sampler=False, num_workers=1, shuffle=False, drop_last=False, cycle=False):
    def _cycle(loader):
        while True:
            for element in loader:
                yield element
            random.shuffle(loader.dataset.items)

    dataset = CSVSet(csv_path)
    loader = data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)

    if cycle:
        loader = _cycle(loader)

    return loader
