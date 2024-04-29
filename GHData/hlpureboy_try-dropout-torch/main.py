import torch
import torch.nn as nn
from models import OverFitFC
from config import *
from dataset import MeMnist
import json
from analysis import models_all, models_plt

# 判定GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = MeMnist(dataset_path, 'train')
test_dataset = MeMnist(dataset_path, 'test')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 损失函数和优化算法
criterion = nn.CrossEntropyLoss()


# 训练函数
def train(model1, name):
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    train_loss_ls = []
    val_loss_ls = []
    train_acc = []
    val_acc = []
    # 训练模型
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model1.train()
        train_loss = 0.0
        val_loss = 0.0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device).to(torch.float32)
            labels = labels.to(device).type(torch.LongTensor)
            # 前向传播和计算loss
            outputs = model1(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # 后向传播和调整参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 每100个batch打印一次数据
            print(' Model {} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(name, epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        train_loss_ls.append((train_loss / len(train_dataset)) * batch_size)
        train_acc.append((correct / len(train_dataset)) * 100)
        model1.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device).to(torch.float32)
                labels = labels.to(device).type(torch.LongTensor)
                outputs = model1(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()
            print(' Model {} Accuracy  test images: {:.4f} %'.format(name, 100 * correct / total))
            print('Model {} Epoch [{}/{}], val , Loss: {:.4f}'.format(name, epoch + 1, num_epochs,
                                                                      val_loss / len(test_dataset)))
            val_loss_ls.append((val_loss / len(test_dataset)) * batch_size)
            val_acc.append(100 * correct / total)
    return {'train_loss': train_loss_ls, 'val_loss': val_loss_ls, 'train_acc': train_acc, 'val_acc': val_acc}


# 定义模型
baseline_model = OverFitFC(baselines)
increase_model = OverFitFC(baselines, increase_dropouts)
reduce_model = OverFitFC(baselines, reduce_dropouts)
avg_model = OverFitFC(baselines, avg_dropouts)

base_result = train(baseline_model, 'baseline')

increase_result = train(increase_model, 'increase')

reduce_result = train(reduce_model, 'reduce')

avg_result = train(avg_model, 'avg')

result = dict()
result['base_result'] = base_result
result['increase_result'] = increase_result
result['reduce_result'] = reduce_result
result['avg_result'] = avg_result

with open('{}_{}.txt'.format(datset_name, num_epochs), 'w') as f:
    f.write(json.dumps(result, ensure_ascii=False, indent=2))

models_all(result, '{}-epoch-{}'.format(datset_name, num_epochs))
models_plt(result, '{}-epoch-{}'.format(datset_name, num_epochs))
