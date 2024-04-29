from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# 训练一个模型分类蜜蜂和蚂蚁,使用迁移学习
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪一个area再resize
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # 把小于0的转化为0，大于1的转化为0
    plt.imshow(inp)  # plt的show传入的是numpy形式的图片
    if title is not None:
        plt.title(title)
    plt.show()
    plt.pause(0.001)


data_dir = 'hymenoptera_data'
# 这里会自动在路径上加上/
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

inputs, classes = next(iter(dataloaders['train']))
print(inputs.shape)
print(classes.shape)
out = torchvision.utils.make_grid(inputs)  # 将多张图片拼接成一张图片
print(out.shape)
imshow(out, title=[class_names[x] for x in classes])

'''
传入一个模型，然后在这个模型上进行修改，训练
'''
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()  # 记录开始时间
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('#' * 10)
        # 每个epoch都由一个训练和验证的过程
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 训练模式
                # 训练模式启用BatchNormalization 和 Dropout
                # 验证模式不启用 BatchNormalization 和 Dropout
            else:
                model.eval()  # 验证模式
            running_loss = 0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):  # 当开启训练模式的时候，才会计算梯度
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.shape[0]  # 这里不能用size了，会出错。针对损失函数，这里要乘上inputs的size
                running_corrects += torch.sum(preds == labels)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('GTrain complete in {:.0f} m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc:{:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


'''
可视化模型的预测结果
该模型通用，用于展示少量预测图片
'''


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()  # 验证
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predict:{}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)


# 加载预训练模型病重置最终完全连接层
model_ft = models.resnet18(pretrained=True)  # 已经训练好的模型
num_ftrs = model_ft.fc.in_features  # 相当于是最后一个层改变一下
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft,
                       criterion,
                       optimizer_ft,
                       exp_lr_scheduler,
                       num_epochs=25)
