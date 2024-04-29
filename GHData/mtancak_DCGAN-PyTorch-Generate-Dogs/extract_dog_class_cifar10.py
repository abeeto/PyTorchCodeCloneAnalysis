import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm

output_class = 5  # dogs = class 5

work_dir = "."
in_sub_dir = "data"
out_sub_dir = "dogs"

if __name__ == "__main__":
    if not os.path.isdir(os.path.join(work_dir, out_sub_dir)):
        os.mkdir(os.path.join(work_dir, out_sub_dir))

    in_dir = os.path.join(work_dir, in_sub_dir)

    train_set = torchvision.datasets.CIFAR10(root=in_dir, train=True,
                                             download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=False, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root=in_dir, train=False,
                                            download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, num_workers=2)

    counter = 0
    for data, label in tqdm(train_loader):
        if label.item() == output_class:
            out_path_file_name = os.path.join(work_dir, out_sub_dir, str(counter) + ".npy")
            np.save(out_path_file_name, data.numpy())
            counter += 1

    for data, label in tqdm(test_loader):
        if label.item() == output_class:
            out_path_file_name = os.path.join(work_dir, out_sub_dir, str(counter) + ".npy")
            np.save(out_path_file_name, data.numpy())
            counter += 1