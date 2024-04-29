# 第七节，迁移学习
# "https://download.pytorch.org/models/resnet18-5c106cde.pth" 这个权重文件提前下好放在data文件夹里
# Author: Sasank Chilamkurthy
# Duplicator: Dy_gs

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def show_pic(tensor, title=None):
    pic = tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pic = std*pic + mean
    pic = np.clip(pic, 0, 1)
    if title is not None:
        plt.title(title)
    plt.imshow(pic)
    plt.pause(0.001)  # 不设置这句pause居然不显示图片了


def train(model, criterion, optimizer, schedule, num_epoch=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0

    for epoch in range(num_epoch):
        print('epoch {}/{}'.format(epoch+1, num_epoch))
        print('-'*10)
        for phase in tmp_li:
            if phase == 'train':
                schedule.step()  # schedule是什么，下面有解释
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0.0

            for data, label in data_loader[phase]:
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(data)
                    _, pre = torch.max(output, 1)
                    # print(pre.shape, label.shape)
                    loss = criterion(output, label)  # 这里为什么不是pre而是output？

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * data.size(0)
                running_correct += torch.sum(pre == label.data)

            epoch_loss = running_loss/dataset_size[phase]
            epoch_score = running_correct.double()/dataset_size[phase]

            print('{} loss : {:.4f}    score : {:.4f}'.format(phase, epoch_loss, epoch_score))

            if phase == 'val' and epoch_score > best_score:
                best_score = epoch_score
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapse = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapse//60, time_elapse % 60))
    print('Best val Score: {:.4f} '.format(best_score))

    model.load(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    # to display some predictions
    was_training = model.traning
    model.eval()
    image_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i, (inpu, label) in enumerate(data_loader['val']):
            inpu = inpu.to(device)
            output = model(inpu)
            _, pre = torch.max(output, 1)

            for j in range(inpu.size()[0]):
                image_so_far += 1
                sub_p = plt.subplot(num_images//2, 2, image_so_far)
                plt.axis('off')
                sub_p.set_titile('predict : {}'.format(class_names[pre[j]]))
                show_pic(inpu.cpu().data[j])  # 这里突然冒出的data属性，让我惊慌

                if image_so_far == num_images:
                    model.tarin(mode=was_training)
                    return
        model.tran(mode=was_training)


if __name__ == '__main__':
    plt.ion()  # 交互模式，开启！
    print('-'*15, 'Start', time.ctime(), '-'*15, '\n')

    data_tran = {
        'train': transforms.Compose([  # 这里将好几个图片预处理操作组合起来
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),  # 这里和训练数据处理的不一样
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_dir = './data/hymenoptera_data/'
    # 这个作者很喜欢用一行代码代替多行代码，下面连着三行都是这样
    tmp_li = ['train', 'val']
    image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_tran[x]) for x in tmp_li}
    # print(image_dataset)
    data_loader = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in tmp_li}
    dataset_size = {x: len(image_dataset[x]) for x in tmp_li}
    class_names = image_dataset[tmp_li[0]].classes

    device = torch.device('cpu')
    # device = torch.device('cuda:0')  # 使用GPU

    # # 展示一小批的图片
    # train_data, train_label = next(iter(data_loader[tmp_li[0]]))
    # out = torchvision.utils.make_grid(train_data)  # 做成几个小格子以便plt.imshow
    # show_pic(out, [class_names[x] for x in train_label])

    # 微调！
    weight_path = './data/resnet18-5c106cde.pth'
    model_ft = models.resnet18()
    model_ft.load_state_dict(torch.load(weight_path))
    # print(time.ctime())
    # # 以下这两行会阻止所有层的后向传播，但接下来更改的最后一层是可以被后向传播的。
    # # 因为初始的requires_grad=True。所以取消下面两行注释后，最后一层的参数会随着训练更新。
    # for param in model_ft.parameters():
    #     param.requires_grad = False

    # 把最后的全连接层改一哈
    num_feature = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_feature, 2)
    model_ft = model_ft.to(device)
    cri = nn.CrossEntropyLoss()
    optimizer_ft = opt.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs，原来schedule是干这个的
    exp_lr_schedule = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train(model_ft, cri, optimizer_ft, exp_lr_schedule)

    visualize_model(model_ft)
    plt.ioff()

    print('%s%s %s %s %s' % ('\n', '-'*16, 'End', time.ctime(), '-'*16))
