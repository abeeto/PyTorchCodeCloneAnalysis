import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)

EPOCH = 5
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# print(train_data.train_data.size())
# print(train_data.test_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.test_labels[0])
# plt.show()

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input shape (1, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),  # output shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2)  # output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = cnn(b_x)
        batch_loss = loss(output, b_y)
        batch_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = cnn(test_x)
            # torch.max选择最大的
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # accuracy = (pred_y == test_y).sum() / float(test_y.size(0))
            accuracy = sum((pred_y == test_y)) / test_y.size(0)
            print('Epoch:%d/%d | train loss: %.4f | test accuracy: %.2f' % (epoch, EPOCH, batch_loss.item(), accuracy))


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
# 保存网络
# torch.save(cnn, 'net.pkl')