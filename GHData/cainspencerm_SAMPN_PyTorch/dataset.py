import torch
from torch.utils import data
import numpy as np
# import random


class DGLDataset(data.Dataset):

    # Share shuffle permutation between all dataloaders.
    _permutation = None

    def __init__(self, data_file, split='full', shuffle=True):
        super().__init__()

        with open(data_file, 'r') as f:
            lines = f.readlines()[1:]  # skip header

        self.smiles = []
        self.labels = []
        for line in lines:
            line = line.strip()
            smile, label = line.split(',')
            if label == 'nan' or label == '' or label == 'None' or \
                smile == 'nan' or smile == '' or smile == 'None':
                continue
                
            self.smiles.append(smile)
            self.labels.append(float(label))

        # Shuffle the data.
        if shuffle:
            if DGLDataset._permutation is None:
                DGLDataset._permutation = np.random.permutation(len(self.labels))
            self.smiles = np.array(self.smiles)[DGLDataset._permutation]
            self.labels = np.array(self.labels, dtype=np.float32)[DGLDataset._permutation]

        num_samples = len(self.labels)

        if split == 'full':
            # No changes necessary.
            pass
        elif split == 'train':
            self.smiles = self.smiles[:int(num_samples * 0.8)]
            self.labels = self.labels[:int(num_samples * 0.8)]
        elif split == 'valid':
            self.smiles = self.smiles[int(num_samples * 0.8):int(num_samples * 0.9)]
            self.labels = self.labels[int(num_samples * 0.8):int(num_samples * 0.9)]
        elif split == 'test':
            self.smiles = self.smiles[int(num_samples * 0.9):]
            self.labels = self.labels[int(num_samples * 0.9):]
        else:
            raise ValueError('Unknown split: {}'.format(split))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        label = self.labels[idx]

        return smile, torch.tensor(label)

    def reshuffle(self):
        DGLDataset._permutation = np.random.permutation(len(self.labels))
        self.smiles = np.array(self.smiles)[DGLDataset._permutation]
        self.labels = np.array(self.labels, dtype=np.float32)[DGLDataset._permutation]
