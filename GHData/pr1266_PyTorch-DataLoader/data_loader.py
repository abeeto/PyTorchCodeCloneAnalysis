import torch
import numpy as np
import os
os.system('cls')

class DataLoader:

    def __init__(self, data, target, batch_size = 16, shuffle = True):
        self.index = 0
        if shuffle == True:
            perm = np.random.permutation(len(data))
            data = data[perm]
            target = target[perm]
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = data.shape[0]
        self.sample_shape = np.array((1,) + data.shape[1:])        
        self.prepare()

    def split_to_batch(self):
        data_array = []
        target_array = []
        for i in range(0, self.n_samples, self.batch_size):            
            step = self.batch_size
            if i + self.batch_size > self.n_samples:
                step = self.n_samples - i
            data_array.append(np.array(self.data[i:i+step], dtype = np.float32))
            target_array.append(np.array(self.target[i:i+step], dtype = np.float32))
        data_array = np.array(data_array, dtype = object)
        target_array = np.array(target_array, dtype = object)
        return data_array, target_array

    def padding(self):
        for index, i in enumerate(self.data_array):
            if i.shape[0] != self.batch_size:
                print(self.target_array[index])
                for j in range(i.shape[0], self.batch_size):
                    padd_data = np.zeros(self.sample_shape)
                    padd_target = np.zeros((1, 1))
                    self.data_array[index] = np.concatenate((self.data_array[index], padd_data))
                    self.target_array[index] = np.concatenate((self.target_array[index], padd_target))
        new_data = []
        new_target = []
        for i in self.data_array:
            new_data.append(i)
        for i in self.target_array:
            new_target.append(i)

        self.data_array = np.array(new_data)
        self.target_array = np.array(new_target)
        return self.data_array, self.target_array

    def prepare(self):        
        self.data_array, self.target_array = self.split_to_batch()
        self.padded_data, self.padded_target = self.padding()
        final_size = (self.padded_data.shape[0], self.batch_size) + tuple(self.sample_shape[1:])
        self.tensor_data = torch.from_numpy(self.padded_data.reshape(final_size))
        self.tensor_target = torch.from_numpy(self.padded_target.reshape((self.padded_data.shape[0], self.batch_size)))
        self.tensor_target = self.tensor_target.view(self.tensor_target.shape + (1,))
        self.data_len = self.padded_data.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 0
        if self.index > self.data_len - 1:
            raise StopIteration
        return self.tensor_data[self.index], self.tensor_target[self.index]

    def __getitem__(self, index):
        return self.tensor_data[index], self.tensor_target[index]

    def __len__(self):
        return self.data_len

x = np.random.rand(100, 32, 32, 3)
y = np.random.rand(100, 1)
d = iter(DataLoader(x, y))
print(len(d))
print(next(d))

        