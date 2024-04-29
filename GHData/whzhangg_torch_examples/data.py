import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class Mydata(Dataset):
    def __init__(self, numpydata):
        normalized = numpydata[:, 1:]
        mean = np.mean(normalized , axis = 0)
        var = np.var(normalized, axis = 0)
        normalized[:] -= mean
        normalized[:] /= np.sqrt(var)

        self.life = torch.from_numpy(numpydata[:, 0]).float().unsqueeze_(1)
        self.others = torch.from_numpy(normalized).float()
        self.nsample, self.nfeature = self.others.shape

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        return self.others[idx], self.life[idx]

def get_life_expectance_data(batchsize = 32):
    pd_data = pd.read_csv("/home/wenhao/python/torch-examples/data/Life_Expectancy.csv")
    pd_data.drop("Year", axis = 1, inplace = True)
    pd_data.drop("Country", axis = 1, inplace = True)
    pd_data.drop("Status", axis = 1, inplace = True)
    pd_data.dropna(inplace=True)
    np_data = pd_data.to_numpy(dtype = float)

    nsample, nfeature = np_data.shape
    np.random.shuffle(np_data)

    all_data = Mydata(np_data)

    test, train = random_split(all_data, [100, nsample - 100], generator=torch.Generator().manual_seed(424242))

    train_dataloader = DataLoader(train, batchsize)
    test_dataloader = DataLoader(test, batchsize)

    return train_dataloader, test_dataloader, (all_data.nfeature, 1)

if __name__ == "__main__":
    train, test = get_life_expectance_data()
    for x,y in train:
        print(x.shape)
        print(y.shape)
        break