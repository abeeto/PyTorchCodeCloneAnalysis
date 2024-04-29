import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class DiabeteDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt(fname = 'data-diabetes.csv', delimiter = ',', dtype = np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0: -1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]


dataset = DiabeteDataset()

train_loader = DataLoader(dataset = dataset, shuffle = True, batch_size = 32, num_workers = 2)

total_epoch = 2

for epoch in range(total_epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs)
        labels = Variable(labels)
        print(epoch, i, "inputs", inputs.data, "labels", labels.data)
