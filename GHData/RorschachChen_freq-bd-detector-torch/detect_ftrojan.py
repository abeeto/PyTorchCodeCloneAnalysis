import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.cnn import Detector
from utils import poison_frequency

device = torch.device('cuda')
model = Detector()
model.load_state_dict(torch.load('detector/6_CNN_CIFAR10.pth'))
model.to(device)

param = {
    "dataset": "CIFAR10",  # CIFAR10
    "target_label": 8,  # target label
    "poisoning_rate": 0.05,  # ratio of poisoned samples
    "label_dim": 10,
    "channel_list": [1, 2],  # [0,1,2] means YUV channels, [1,2] means UV channels
    "magnitude": 30,
    "YUV": True,
    "window_size": 32,
    "pos_list": [(31, 31), (15, 15)],
}
from keras.datasets import cifar10

(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype(np.float) / 255.
x_test_pos = poison_frequency(x_test.copy(), y_test.copy(), param)
y_test_pos = np.array([[param['target_label']]] * x_test_pos.shape[0], dtype=np.long)

x_dct_test = np.vstack((x_test, x_test_pos))  # [:,:,:,0]
y_dct_test = (np.vstack((np.zeros((x_test.shape[0], 1)), np.ones((x_test.shape[0], 1))))).astype(np.int)

x_dct_test, x_test_pos = np.transpose(x_dct_test, (0, 3, 1, 2)), np.transpose(
    x_test_pos, (0, 3, 1, 2))
x_test_pos, y_test_pos = torch.tensor(x_test_pos, dtype=torch.float), torch.tensor(y_test_pos,
                                                                                   dtype=torch.long).view(
    (-1,))

x_dct_test, y_dct_test = torch.tensor(x_dct_test, dtype=torch.float), torch.tensor(y_dct_test,
                                                                                   dtype=torch.long).view(
    (-1,))

only_poison = DataLoader(TensorDataset(x_test_pos, y_test_pos), batch_size=64, shuffle=False)
clean_and_poison = DataLoader(TensorDataset(x_dct_test, y_dct_test), batch_size=64, shuffle=False)

model.eval()
acc_meter = 0
runcount = 0
print(f'size: {len(only_poison.dataset)}')
with torch.no_grad():
    for load in only_poison:
        d = load[0].to(device)
        t = load[1].to(device)
        pred = model(d)
        pred = pred.max(1, keepdim=True)[1]
        acc_meter += pred.eq(t.view_as(pred)).sum().item()
        runcount += d.size(0)
print(f'only poison detect success rate: {100 * acc_meter / runcount}')

acc_meter = 0
runcount = 0
print(f'size: {len(clean_and_poison.dataset)}')
with torch.no_grad():
    for load in clean_and_poison:
        d = load[0].to(device)
        t = load[1].to(device)
        pred = model(d)
        pred = pred.max(1, keepdim=True)[1]
        acc_meter += pred.eq(t.view_as(pred)).sum().item()
        runcount += d.size(0)
print(f'mixture detect success rate: {100 * acc_meter / runcount}')
