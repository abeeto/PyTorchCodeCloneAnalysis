import torch.optim as optim
import torch.nn as nn
from testMnistLoss import Net

criterion = nn.CrossEntropyLoss()
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#train
dataloader = 0 #为了防止报错 放在这
inputs = 0 #为了防止报错
labels = 0 #为了防止报错
for epoch in range(2):
    running_loss = 0.0
    for data in enumerate(dataloader):
        optimizer.zero_grad()

        # forward + backward + optimize 总的来说，训练就是四步：跑网络 算损失 反向传播 使用优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
