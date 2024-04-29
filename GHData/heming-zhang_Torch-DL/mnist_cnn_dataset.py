import numpy as np
from torch.utils.data import Dataset

# custom yourself class dataset for dataloader
class MnistCNNTrain(Dataset):
    def __init__(self):
        train_img = np.load("./mnist/train_image.npy")
        train_label = np.load("./mnist/train_label.npy")
        self.x = np.reshape(train_img, (60000, 1, 28, 28))
        self.y = np.array(train_label)
        self.len = np.shape(self.x)[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

class MnistCNNTest(Dataset):
    def __init__(self):
        test_img = np.load("./mnist/test_image.npy")
        test_label = np.load("./mnist/test_label.npy")
        self.x = np.reshape(test_img, (10000, 1, 28, 28))
        self.y = np.array(test_label)
        self.len = np.shape(self.x)[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len