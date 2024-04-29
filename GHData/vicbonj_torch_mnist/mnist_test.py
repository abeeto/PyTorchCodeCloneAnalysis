from net.networks import LeNet5
import numpy as np
import keras
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 1, 28, 28)).astype('float32')
x_test = x_test.reshape((-1, 1, 28, 28)).astype('float32')
x_train /= 255.
x_test /= 255.
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

class TMPDataset(Dataset):
    def __init__(self, a, b):
        self.x = a
        self.y = b

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.y)

batch_size = 256
    
data_train_loader = DataLoader(TMPDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=1)
data_test_loader = DataLoader(TMPDataset(x_test, y_test), batch_size=batch_size, num_workers=1)

x_train_t = torch.from_numpy(x_train).cuda()
y_train_t = torch.from_numpy(y_train).cuda()

net = LeNet5().cuda()
#if torch.cuda.device_count() > 1:
#    net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), eps=1e-7)

epochs = 5

#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True

#scaler = GradScaler()

for e in range(epochs):
    print('Epoch {}'.format(e))
    
    '''
    for i, (images, labels) in tqdm(enumerate(data_train_loader)):
        images = images.cuda()
        labels = labels.cuda()
        #images = torch.autograd.Variable(images.cuda())
        #labels = torch.autograd.Variable(labels.cuda())
        
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    '''
    
    for i in tqdm(range(int(len(x_train_t)/batch_size))):
        images = x_train_t[int(i*batch_size):int(i*batch_size+batch_size)]
        labels = y_train_t[int(i*batch_size):int(i*batch_size+batch_size)]
        optimizer.zero_grad()
        with autocast():
            output = net(images)
            loss = criterion(output, labels)
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        loss.backward()
        optimizer.step()
    
