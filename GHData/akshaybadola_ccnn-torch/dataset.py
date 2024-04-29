import numpy as np
from pathlib import Path
import pickle

from PIL import Image
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    def __init__(self, images, targets, indicator_vectors,
                 in_memory, transform=None):
        self.images = images
        self.targets = targets
        self.indicator_vectors = indicator_vectors
        self.in_memory = in_memory
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        target, ind_v = self.targets[i], self.indicator_vectors[i]
        if self.in_memory:
            img = self.images[i]
        else:
            fp = Image.open(self.images[i])
            img = np.array(fp)
            fp.close()
        if self.transform:
            return self.transform(img), target, ind_v
        else:
            return img, target, ind_v


class CUBData:
    """CUB 2011 Birds dataset.

    Args:
        root_folder: Root Images folder. Each subfolder is assumed to be a bird
                     category with corresponding images.
        classes_file: A space separated file of class number and bird name
                      The bird name may further be in a {num}.{str} format
        indicator_vectors_file: A pickle file containing the indicator vectors
                                corresponding to each image
        in_memory: Whether to load all the images in memory.
                   Can speed up image loading later at thee expense of memory.
                   The system should have at least 32GB of RAM for ths
                   option to be enabled.

    The images are read and indexed at :code:`__init__`. Every 10th image is set aside
    for validation.

    Use :meth:`get_model` to actually get the dataset.

    """
    def __init__(self, root_folder: Path, classes_file: Path,
                 indicator_vectors_file: Path,
                 in_memory: bool = False):
        self._data = []
        with open(indicator_vectors_file, "rb") as f:
            self.indicator_vectors = pickle.load(f)
        self.example_vector = self.indicator_vectors.values().__iter__().__next__()
        with open(classes_file) as f:                      # type: ignore
            classes = filter(None, f.read().split("\n"))   # type: ignore
        classes = [c.split() for c in classes]           # type: ignore
        self._classes = {c[1]: int(c[0]) for c in classes}  # type: ignore
        self._labels = {k.split(".")[1]: v for k, v in self._classes.items()}  # type: ignore
        train_inds = []
        val_inds = []
        ind = 0
        for dir in root_folder.iterdir():
            label = self._labels[dir.name]
            for i, img in enumerate(dir.iterdir()):
                self._data.append((img.absolute(), label))
                if i % 10:
                    train_inds.append(ind)
                else:
                    val_inds.append(ind)
        self.images, self.targets = zip(*self._data)
        self._fnames = [f.name for f in self.images]
        self.in_memory = in_memory
        if self.in_memory:
            _images = []
            for img in self.images:
                fp = Image.open(img)
                _images.append(np.array(fp))
                fp.close()
            self.images = _images  # type: ignore
        self.train_inds = train_inds
        self.val_inds = val_inds

    def get_data(self, split, transform=None):
        """Get CUB data for :code:`split`.

        Split can be "train" or "val"

        Args:
            split: train or val
            transform: :code:`torchvision.transforms.transform` that will be applied to each
                       data poin
        """
        if split == "train":
            images = [self.images[i] for i in self.train_inds]
            targets = [self.targets[i] for i in self.train_inds]
            ind_vs = [self.indicator_vectors[self._fnames[i]] for i in self.train_inds]
        elif split == "val":
            images = [self.images[i] for i in self.val_inds]
            targets = [self.targets[i] for i in self.val_inds]
            ind_vs = [np.ones_like(self.example_vector) for _ in self.val_inds]
        return CUBDataset(images, targets, ind_vs, self.in_memory, transform)
