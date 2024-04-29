import torch
from torch.utils.data import Dataset
import utils.DataPreProc as DPP
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, DataRoot, Train = True):
        super().__init__()
        self.dataroot = DataRoot
        datas = []
        self.filename = 'train.txt'
        if not Train:
            self.filename = 'validate.txt'
        with open(os.path.join(DataRoot, self.filename), 'r', encoding='utf-8') as f:
            for i in f.read().splitlines():
                datas.append(i)
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # todo
        voxel = DPP.binvox2Tensor('{}/datas/{}/{}.binvox'.format(self.dataroot, self.datas[index],  self.datas[index])).cuda()
        pointCloud = DPP.getPointCloud('{}/datas/{}/{}.pcd'.format(self.dataroot, self.datas[index], self.datas[index])).cuda()
        closest = np.load('{}/datas/{}/{}.npy'.format(self.dataroot, self.datas[index], self.datas[index]))
        closest = torch.tensor(closest, dtype = torch.float32).cuda()
        target = {
            'voxel':voxel,
            'points':pointCloud,
            'closest':closest,
        }
        return target