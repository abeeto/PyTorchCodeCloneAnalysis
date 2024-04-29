import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
from models.cnn import Detector
from utils import patching_test, dct2

device = torch.device('cuda')
model = Detector()
model.load_state_dict(torch.load('detector/6_CNN_CIFAR10.pth'))
model.to(device)

from keras.datasets import cifar10

(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype(np.float) / 255.

poi_test = np.zeros_like(x_test)
attack_list = {'l2_inv'}
for i in range(x_test.shape[0]):
    attack_name = random.sample(attack_list, 1)[0]
    poi_test[i] = patching_test(x_test[i], attack_name)

x_dct_test = np.vstack((x_test, poi_test))  # [:,:,:,0]
y_dct_test = (np.vstack((np.zeros((x_test.shape[0], 1)), np.ones((x_test.shape[0], 1))))).astype(np.int)
for i in range(x_dct_test.shape[0]):
    for channel in range(3):
        x_dct_test[i][:, :, channel] = dct2((x_dct_test[i][:, :, channel] * 255).astype(np.uint8))

x_dct_poi, y_dct_poi = x_dct_test[10000:], y_dct_test[10000:]
x_dct_poi = np.transpose(x_dct_poi, (0, 3, 1, 2))
x_dct_poi, y_dct_poi = torch.tensor(x_dct_poi, dtype=torch.float), torch.tensor(y_dct_poi,
                                                                                dtype=torch.long).view(
    (-1,))
only_poison = DataLoader(TensorDataset(x_dct_poi, y_dct_poi), batch_size=64, shuffle=False)

x_dct_test = np.transpose(x_dct_test, (0, 3, 1, 2))
x_dct_test, y_dct_test = torch.tensor(x_dct_test, dtype=torch.float), torch.tensor(y_dct_test,
                                                                                   dtype=torch.long).view(
    (-1,))
clean_and_poison = DataLoader(TensorDataset(x_dct_test, y_dct_test), batch_size=64, shuffle=False)

model.eval()
acc_meter = 0
runcount = 0
with torch.no_grad():
    for load in clean_and_poison:
        d = load[0].to(device)
        t = load[1].to(device)
        pred = model(d)
        pred = pred.max(1, keepdim=True)[1]
        acc_meter += pred.eq(t.view_as(pred)).sum().item()
        runcount += d.size(0)
print(f'mixture detect success rate: {100 * acc_meter / runcount}')

acc_meter = 0
runcount = 0
with torch.no_grad():
    for load in only_poison:
        d = load[0].to(device)
        t = load[1].to(device)
        pred = model(d)
        pred = pred.max(1, keepdim=True)[1]
        acc_meter += pred.eq(t.view_as(pred)).sum().item()
        runcount += d.size(0)
print(f'only poi success rate: {100 * acc_meter / runcount}')
