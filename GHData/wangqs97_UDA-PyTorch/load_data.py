from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from autoaugment import CIFAR10Policy
from PIL import Image
import pickle
import os


class SupImgDataSet(torchvision.datasets.CIFAR10):
    def __init__(self, root, index, transform=None, download=True):
        super().__init__(root, transform=transform, download=download)
        self.data = self.data[index]
        self.targets = np.asarray(self.targets)[index]


class UnSupImgDataSet(torchvision.datasets.CIFAR10):
    def __init__(self, root, index, transform=None, target_transform=None, download=True):
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.data = self.data[index]
        self.targets = self.data

    def __getitem__(self, index):
        aug_img, img = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        aug_img = Image.fromarray(aug_img)
        if self.transform is not None:
            aug_img = self.transform(aug_img)
        if self.target_transform is not None:
            img = self.target_transform(img)
        return aug_img, img


def sup_unsup_proc(label, sup_per_class, r):
    if r:
        with open ('temp_label.pkl', 'rb') as f:
            a = pickle.load(f)
            return a[0], a[1]
    else:
        label = np.asarray(label)
        sup_idx = []
        unsup_idx = []
        for i in range(10):
            idx = np.where(label == i)[0]
            np.random.shuffle(idx)
            sup_idx.extend(idx[: sup_per_class])
            unsup_idx.extend(idx[sup_per_class:])
        with open ('temp_label.pkl', 'wb') as f:
            pickle.dump([sup_idx, unsup_idx], f)
        return sup_idx, unsup_idx


def dataload(r, supnum=4000):
    root = 'data/'
    if not os.path.exists(root):
        os.mkdir(root)
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                                 std=[0.24703223, 0.24348513, 0.26158784])])
    aug_t = transforms.Compose([transforms.RandomHorizontalFlip(),
                                CIFAR10Policy(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                                     std=[0.24703223, 0.24348513, 0.26158784])])
    base_set = torchvision.datasets.CIFAR10(root=root)
    sup_idx, unsup_idx = sup_unsup_proc(base_set.targets, int(supnum / 10), r)
    sup_data = SupImgDataSet(root=root, index=sup_idx, transform=t)
    val_data = torchvision.datasets.CIFAR10(root=root, train=False, transform=t)
    unsup_data = UnSupImgDataSet(root=root, index=unsup_idx, transform=aug_t, target_transform=t)
    sup_dataloader = DataLoader(dataset=sup_data, batch_size=64, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=16, num_workers=2, shuffle=True)
    unsup_dataloader = DataLoader(dataset=unsup_data, batch_size=128, num_workers=2, shuffle=True)
    return sup_dataloader, val_dataloader, unsup_dataloader


if __name__ == "__main__":
    sup_dataloader, val_dataloader, unsup_dataloader = dataload(0)