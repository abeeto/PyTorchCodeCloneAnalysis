import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

# writer = SummaryWriter(comment='_normal_conv')
writer = SummaryWriter(comment='_se-block-r=2')

NUM_TRAINING_SAMPLES = 50000
EPOCHS = 500
learning_rate = 1e-4

SEED = 256
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
dataset = torchvision.datasets.FashionMNIST(root='.')

# 生成随机索引
order = np.argsort(np.random.random(dataset.train_labels.shape))
print(order)
# 打乱数据集中样本顺序
data = dataset.train_data[order].float()
data = data.float()
data = torch.unsqueeze(data, 1)
target = dataset.train_labels[order]

data = data.to(device)
target = target.to(device)
D_IN = 1
D_OUT = 10

#指定训练集和测试集
data_tr = data[0:NUM_TRAINING_SAMPLES]
data_ts = data[NUM_TRAINING_SAMPLES:]
target_tr = target[0:NUM_TRAINING_SAMPLES]
target_ts = target[NUM_TRAINING_SAMPLES:]

class FMNIST_data(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)
    
    def getfeatures(self, index):
        return self.x[index]
    
    def getprice(self, index):
        return self.y[index]

FMNIST_tr = FMNIST_data(data_tr, target_tr)
FMNIST_ts = FMNIST_data(data_ts, target_ts)

loader_tr = torch.utils.data.DataLoader(dataset=FMNIST_tr, batch_size=1024, shuffle=True)
loader_ts = torch.utils.data.DataLoader(dataset=FMNIST_ts, batch_size=1024, shuffle=False)

# an example of building a CNN model on PyTorch
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

class SE_block(nn.Module):
    def __init__(self, D_IN, D_OUT, r=16):
        super(SE_block, self).__init__()
        self.D_IN = D_IN
        self.D_OUT = D_OUT
        self.conv1 = nn.Conv2d(D_IN, D_OUT, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(D_OUT)
        self.fc1 = nn.Linear(D_OUT, int(D_OUT/r))
        # self.fc1_bn = nn.BatchNorm2d(int(D_OUT/r))
        self.fc2 = nn.Linear(int(D_OUT/r), D_OUT)
        # self.fc2_bn = nn.BatchNorm2d(int(D_OUT/r))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x_keep = x
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = self.fc1(x.view(-1, self.D_OUT))
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x_keep * x.view(-1, self.D_OUT, 1, 1)


class FMNIST_conv(nn.Module):
    def __init__(self, D_IN, D_OUT):
        super(FMNIST_conv, self).__init__()
        self.conv1 = nn.Conv2d(D_IN, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.dense1 = nn.Linear(22*22*32, 64)
        self.dense2 = nn.Linear(64, 10)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(-1, 22*22*32)
        out = F.relu(self.dense1(out))
        out = self.dense2(out)
        return out
        
class FMNIST_conv_(nn.Module):
    def __init__(self, D_IN, D_OUT):
        super(FMNIST_conv_, self).__init__()
        self.conv1 = nn.Conv2d(D_IN, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dense1 = nn.Linear(7*7*32, 64)
        self.dense2 = nn.Linear(64, 10)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = out.view(-1, 7*7*32)
        out = F.relu(self.dense1(out))
        out = self.dense2(out)
        return out

class FMNIST_conv_SE(nn.Module):
    def __init__(self, D_IN, D_OUT):
        super(FMNIST_conv_SE, self).__init__()
        self.conv1 = SE_block(D_IN, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = SE_block(32, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dense1 = nn.Linear(7*7*32, 64)
        self.dense2 = nn.Linear(64, 10)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = out.view(-1, 7*7*32)
        out = F.relu(self.dense1(out))
        out = self.dense2(out)
        return out

dataset = {'train': FMNIST_tr, 'val': FMNIST_ts}
dataloader = {'train': loader_tr, 'val': loader_ts}

phase = ['train', 'val']
LOSS = {'train': torch.empty(EPOCHS), 'val': torch.empty(EPOCHS)}
ACCURACY = {'train': torch.empty(EPOCHS), 'val': torch.empty(EPOCHS)}

# model = FMNIST_conv_(D_IN, D_OUT)
model = FMNIST_conv_SE(D_IN, D_OUT)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.2)

for iter in range(EPOCHS):
    print('iter:', iter)
    scheduler.step()
    for ph in phase:
        loss_total = 0
        correct_total = 0
        if ph == 'train':
            model.train()
        elif ph == 'test':
            model.eval()
        for inputs, targets in dataloader[ph]:
            targets_pred = model(inputs)
            loss = criterion(targets_pred, targets)
            loss_total += loss.item() * len(inputs)

            _, targets_pred = torch.max(targets_pred, 1)
            correct_total += torch.sum(targets_pred == targets)
            if ph == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        LOSS[ph][iter] = loss_total / dataset[ph].__len__()
        ACCURACY[ph][iter] = correct_total.float() / dataset[ph].__len__()
        writer.add_scalar('loss/'+ph, LOSS[ph][iter], iter)
        writer.add_scalar('acc/'+ph, ACCURACY[ph][iter] * 100, iter)
        print('{} loss: {:.6f}'.format(ph, LOSS[ph][iter]))
        print('{} acc: {:.6f} %'.format(ph, ACCURACY[ph][iter] * 100))