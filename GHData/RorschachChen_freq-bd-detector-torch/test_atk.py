import numpy as np
import torch
import torchvision.transforms as transforms
from keras.datasets import cifar10
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from models.cnn import cifar_model
from utils import poison_frequency

device = torch.device('cuda')
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

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

(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype(np.float) / 255.
x_test_pos = poison_frequency(x_test.copy(), y_test.copy(), param)
y_test_pos = np.array([[param['target_label']]] * x_test_pos.shape[0], dtype=np.long)
x_test, x_test_pos = np.transpose(x_test, (0, 3, 1, 2)), np.transpose(
    x_test_pos, (0, 3, 1, 2))
x_test_pos, y_test_pos = torch.tensor(x_test_pos, dtype=torch.float), torch.tensor(y_test_pos,
                                                                                   dtype=torch.long).view(
    (-1,))

clean_test = CIFAR10(root='.', train=False, download=True, transform=transform_test)
clean_test_loader = DataLoader(clean_test, batch_size=64, num_workers=0)
poison_test_loader = DataLoader(TensorDataset(x_test_pos, y_test_pos), batch_size=64, shuffle=False)

net = cifar_model().to(device)
net.load_state_dict(torch.load('attacker/cifar.pth'))
criterion = nn.CrossEntropyLoss().cuda()


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
print('{:.4f} \t {:.4f}'.format(po_test_loss, po_test_acc))

cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
print('{:.4f} \t {:.4f}'.format(cl_test_loss, cl_test_acc))
