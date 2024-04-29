import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

NUM_TRAINING_SAMPLES = 50000
EPOCHS = 5
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dataset = torchvision.datasets.FashionMNIST(root='.')

# 生成随机索引
order = np.argsort(np.random.random(dataset.train_labels.shape))

# 打乱数据集中样本顺序
data = dataset.train_data[order].float()
data = data.reshape(data.shape[0], -1).float()
target = dataset.train_labels[order]

data = data.to(device)
target = target.to(device)
D_IN = data.shape[1]
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
loader_ts = torch.utils.data.DataLoader(dataset=FMNIST_ts, batch_size=1024, shuffle=True)

class FMNIST_dense(nn.Module):
    def __init__(self, D_IN, D_OUT):
        super(FMNIST_dense, self).__init__()
        self.inputLayer = nn.Linear(D_IN, 64)
        self.denseLayer1 = nn.Linear(64, 128)
        self.denseLayer2 = nn.Linear(128, 128)
        self.denseLayer3 = nn.Linear(128, 64)
        self.outputLayer = nn.Linear(64, D_OUT)

    def forward(self, x):
        x = F.relu(self.inputLayer(x))
        x = F.relu(self.denseLayer1(x))
        x = F.relu(self.denseLayer2(x))
        x = F.relu(self.denseLayer3(x))
        return self.outputLayer(x)

dataset = {'train': FMNIST_tr, 'val': FMNIST_ts}
dataloader = {'train': loader_tr, 'val': loader_ts}

phase = ['train', 'val']
MSE = {'train': torch.empty(EPOCHS), 'val': torch.empty(EPOCHS)}
ACCURACY = {'train': torch.empty(EPOCHS), 'val': torch.empty(EPOCHS)}
model = FMNIST_dense(D_IN, D_OUT)
model = model.to(device)

# model = torch.nn.Sequential(
#     torch.nn.Linear(D_IN, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, 128),
#     torch.nn.ReLU(),
#     torch.nn.Linear(128, 128),
#     torch.nn.ReLU(),
#     torch.nn.Linear(128, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, D_OUT),
# )
# model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
plt.ion()

for iter in range(EPOCHS):
    lr_scheduler.step()
    print(lr_scheduler.get_lr())
    print('iter:', iter)
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
        MSE[ph][iter] = loss_total / dataset[ph].__len__()
        ACCURACY[ph][iter] = correct_total.float() / dataset[ph].__len__()
        print('{} loss: {:.6f}'.format(ph, MSE[ph][iter]))
        print('{} acc: {:.6f} %'.format(ph, ACCURACY[ph][iter] * 100))

        plt.cla()
        plt.ylim([0,1])
        plt.plot(np.linspace(1, iter+1, iter+1), ACCURACY['train'].detach().numpy()[0: iter+1])
        plt.pause(0.1)
        plt.plot(np.linspace(1, iter+1, iter+1), ACCURACY['val'].detach().numpy()[0: iter+1])
        plt.pause(0.1)
plt.ioff()
plt.show()

plt.figure()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(np.linspace(1,EPOCHS,EPOCHS), ACCURACY['train'].detach().numpy())
plt.plot(np.linspace(1,EPOCHS,EPOCHS), ACCURACY['val'].detach().numpy())
plt.show()

