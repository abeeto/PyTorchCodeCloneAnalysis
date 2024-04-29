from torch.utils.data import Dataset
import os
import rasterio
import numpy as np
import torch


class EurosatDataset(Dataset):
    x = []
    label = []

    def __init__(self, root_dir, transform):
        self.transform = transform
        # -1 cause it starts from the root dir ('/train' or '/test' ;) )
        i = -1
        for dirName, subdirList, fileList in os.walk(root_dir):
            print('\t%s' % dirName)
            for fname in fileList:
                if (fname.endswith(".tif")):
                    with rasterio.open(os.path.join(dirName, fname)) as src:
                        r = src.read(4)
                        g = src.read(3)
                        b = src.read(2)
                        # Numpy Img convention: HxWxC
                        rgb = np.array([r, g, b], dtype=np.uint8).reshape((64, 64, 3))
                    self.x.append(rgb)
                    self.label.append(i)
            i += 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # # Convert image and label to torch tensors
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # img = torch.from_numpy(self.x[idx].transpose((2, 0, 1)))

        if self.transform:
            img = self.transform(self.x[idx])

        # OneHot Labels
        labels = torch.LongTensor(range(0, 10)).view(-1, 1)
        # tensor of 10 elements, the i rows has 1 at the i position, 0 else
        labels_one_hot = torch.zeros(10, 10).scatter_(1, labels, 1)

        return img, labels_one_hot[self.label[idx]]
