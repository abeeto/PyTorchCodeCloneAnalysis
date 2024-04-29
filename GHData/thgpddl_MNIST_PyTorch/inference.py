# -*- encoding: utf-8 -*-
"""
@File    :   inference.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/29 20:23   thgpddl      1.0         None
"""
import torch
from torch import nn
from model import LeNet
from data import getLoader

data_train_loader, data_test_loader = getLoader()

model_path = "outputs/epoch_9_acc_93.95%.pth"
save_info = torch.load(model_path)  # 加载模型的所有信息(状态信息和参数信息)
model = LeNet()  # 定义模型结构

criterion = nn.CrossEntropyLoss()  # 定义损失函数

# 加载参数
# 将state_dict中的parameters和buffers复制到此module和它的后代中
# state_dict中的key必须和 model.state_dict()返回的key一致
model.load_state_dict(save_info['model'])
model.eval()

test_loss = 0
correct = 0
total = 0
with torch.no_grad():  # 不进行梯度更新，这是测试，不是训练
    for batch_idx, (inputs, targets) in enumerate(data_test_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        print(batch_idx, len(data_test_loader),
              "Loss: %.3f | Acc:%.3f%%(%d/%d)" % (test_loss / (batch_idx + 1), 100 * correct / total, correct, total))
