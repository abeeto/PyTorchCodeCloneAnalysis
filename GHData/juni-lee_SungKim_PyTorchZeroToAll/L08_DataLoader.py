"""
If you have 1000 training examples, and your batch size is 500,
then it will take 2 iterations to complete 1 epoch.
"""

import numpy as np
import torch

class DiabetesDataset(torch.utils.data.Dataset):
    """ Diabetes dataset. """

    # download, read data, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # return on item on the index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # return the data length
    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=32,
                                           shuffle=True)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)

        # Run your training process
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')

print()