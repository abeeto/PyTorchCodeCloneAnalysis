import os
import torch.utils.data as data
import numpy as np
import torch

class SenSatDataset(data.Dataset):
    def __init__(self, root='./data/processed', npoints=4096, mode='train'):
        self.npoints = npoints
        self.root = root + '_' + mode
        self.num_classes = 13
        
        self.data_files = [os.path.join(self.root, f) for f in os.listdir(self.root) if 'label' not in f]
        self.label_files = [f.replace('data.npy', 'label.npy') for f in self.data_files]

    def __getitem__(self, index):
        data_name = self.data_files[index]
        label_name = self.label_files[index]

        data = np.load(data_name)
        label = np.load(label_name)
        assert data.shape[0] == label.shape[0]

        choice = np.random.choice(data.shape[0], self.npoints, replace=True)
        _data = data[choice, :]
        _label = label[choice]
        data = torch.from_numpy(_data)
        label = torch.from_numpy(_label)
        return data, label

    def __len__(self):
        return len(self.data_files)

class SenSatTestDataset(data.Dataset):
    def __init__(self, root='./data/processed', npoints=4096, mode='test'):
        self.npoints = npoints
        self.root = root + '_' + mode
        self.num_classes = 13
        
        self.data_files = [os.path.join(self.root, f) for f in os.listdir(self.root) if 'label' not in f]
        self.label_files = [f.replace('data.npy', 'label.npy') for f in self.data_files]

    def __getitem__(self, index):
        data_name = self.data_files[index]
        label_name = self.label_files[index]

        data = np.load(data_name)
        label = np.load(label_name)
        assert data.shape[0] == label.shape[0]

        if data.shape[0] < self.npoints:
            data = np.tile(data, (2, 1))
            data = data[:self.npoints, :]
            batch_data = data.reshape((1, self.npoints, data.shape[1]))
            label = np.tile(label, (2))
            label = label[:self.npoints]
            batch_label = label.reshape((1, self.npoints))

            batch_data = torch.from_numpy(batch_data)
            batch_label = torch.from_numpy(batch_label)
            return batch_data, batch_label
        
        batch_size = data.shape[0] // self.npoints + 1
        batch_data = np.zeros((batch_size, self.npoints, data.shape[1]), dtype=np.float32)
        batch_label = np.zeros((batch_size, self.npoints))

        batch_data[:(batch_size-1), :, :] = data[:(batch_size-1)*self.npoints, :].reshape((batch_size-1, self.npoints, data.shape[1]))
        batch_data[(batch_size-1), :, :] = data[-self.npoints:, :]

        batch_label[:(batch_size-1), :] = label[:(batch_size-1)*self.npoints].reshape((batch_size-1, self.npoints))
        batch_label[(batch_size-1), :] = label[-self.npoints:]

        batch_data = torch.from_numpy(batch_data)
        batch_label = torch.from_numpy(batch_label)
        return batch_data, batch_label

    def __len__(self):
        return len(self.data_files)

class SenSatPredDataset(data.Dataset):
    def __init__(self, root='./data/processed_test', npoints=4096, mode='pred'):
        self.npoints = npoints
        self.root = root
        self.num_classes = 13
        
        self.file_names = [f for f in os.listdir(self.root)]
        self.data_files = [os.path.join(self.root, f) for f in self.file_names]

    def __getitem__(self, index):
        data_name = self.data_files[index]
        file_name = self.file_names[index]
        data = np.load(data_name)

        if data.shape[0] < self.npoints:
            data = np.tile(data, (2, 1))
            data = data[:self.npoints, :]
            batch_data = data.reshape((1, self.npoints, data.shape[1]))

            batch_data = torch.from_numpy(batch_data)
            return batch_data
        
        batch_size = data.shape[0] // self.npoints + 1
        batch_data = np.zeros((batch_size, self.npoints, data.shape[1]), dtype=np.float32)

        batch_data[:(batch_size-1), :, :] = data[:(batch_size-1)*self.npoints, :].reshape((batch_size-1, self.npoints, data.shape[1]))
        batch_data[(batch_size-1), :, :] = data[-self.npoints:, :]

        batch_data = torch.from_numpy(batch_data)
        return batch_data, file_name

    def __len__(self):
        return len(self.data_files)



if __name__ == '__main__':
    data_set = SenSatDataset()
    print(len(data_set))
    v, l = data_set[0]
    print(v.size(), v.type(), l.size(), l.type())