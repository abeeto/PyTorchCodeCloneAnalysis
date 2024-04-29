import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import net  # 自定义的网络模块
import readpic  # 读自己手写的图片


# hyperparameters
batch_size = 128
learning_rate = 1e-2
num_epoches = 5

# 标准化
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = net.LeNet()
if torch.cuda.is_available():
    model = model.cuda()


# 定义loss函数和优化方法
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# for epoch in range(num_epoches):
#     model.train()
#     for data in train_loader:   # 每次取一个batch_size张图片
#         img, label = data   # img.size:128*1*28*28
#         # img = img.view(img.size(0), -1)  # 展开成128 *784（28*28）
#         if torch.cuda.is_available():
#             img = img.cuda()
#             label = label.cuda()
#         output = model(img)
#         loss = loss_fn(output, label)
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print('epoch:', epoch, '|loss:', loss.item())
#     # 在测试集上检验效果
#     model.eval()  # 将模型改为测试模式
#     eval_loss = 0
#     eval_acc = 0
#     for data in test_loader:
#         img, label = data
#         if torch.cuda.is_available():
#             img = img.cuda()
#             label = label.cuda()
#         out = model(img)
#         loss = loss_fn(out, label)
#         # print(label.size(0))
#         eval_loss += loss.item() * label.size(0)   # lable.size(0)=128
#         _, pred = torch.max(out, 1)
#         num_correct = (pred == label).sum()
#         eval_acc += num_correct.item()
#     print('Epoch:{}, Test loss:{:.6f}, Acc:{:.6f}'.format(epoch, eval_loss/(len(test_dataset)), eval_acc/(len(test_dataset))))




# 要是只是训练网络下面的代码可以直接删去

figure = readpic.readImage(path='./3.png', size=28)   # figure dim=[1, 28, 28]
# figure = figure.unsqueeze(0)   # figure dim = [1, 1, 28, 28]
figure = figure.cuda()
y_pred = model(figure)
_, pred = torch.max(y_pred, 1)
print('prediction = ', pred.item())



