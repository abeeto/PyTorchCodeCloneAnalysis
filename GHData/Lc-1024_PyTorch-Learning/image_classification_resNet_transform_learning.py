# CNN模型的迁移学习transform learning
# fine tuning 微调，从一个预训练模型开始，重新训练整个模型
# feature extraction 特征抽取，仅仅更新最后几层网络
# 用datasets和dataloader处理数据
# 用resnet18初始化模型及参数，并改变最后一层的参数
# 用SGD和CrossEntropyLoss训练模型


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import time
import os
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
MOMENTUM = 0.9
# 数据集所在目录
data_dir = ".data\\hymenoptera_data"
# Models to choose [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
# 数据中类别的数量
num_classes = 2
# feature_extract 表示使用fine tuning / feature extraction
feature_extract = True
# 输入图片的尺寸
input_size = 224

'''
# 将图片转换成迭代器后输出
all_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"), 
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]))
loader = torch.utils.data.DataLoader(all_imgs, BATCH_SIZE, shuffle=True, num_workers=0)
img = next(iter(loader))[0]

# 显示图片
unloader = transforms.ToPILImage()
plt.ion()
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

plt.figure()
imshow(img[11], title='Image')
'''
# 不同数据集的不同转换格式
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}
# 生成数据表和迭代器
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ["train", "val"]}

# 将所有的参数的自动求导关闭
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
# 初始化模型 
# num_classes是输出的种类数量
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        # 用resnet18来初始化模型，并且初始化参数
        model_ft = models.resnet18(pretrained=use_pretrained)
        # 如果不是feature extraction则将不需要的参数导数关闭
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        # print(model_ft.fc.weight.requires_grad) # True
    else:
        print("model not implemented.")
        return None, None

    return model_ft, input_size

# 下面是feature extraction的训练
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=5):
    # 在训练之前先保存最优的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    val_acc_history = []
    for e in range(num_epochs):
        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.autograd.set_grad_enabled(phase=="train"):
                    outputs = model(inputs) # batch_size * 2
                    loss = loss_fn(outputs, labels)

                preds = outputs.argmax(dim=1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print("Phase: {} loss: {} acc: {}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

model_ft = model_ft.to(device)
# filter: 过滤器，第一位为判别器/函数，后面为参数，若判别为True则参数能输出，最终输出为iter
# lambda: 简单的函数，输入为p，返回p.requires_grad的值
# 如此一番之后，只有requires_grad的参数才会被处理
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                model_ft.parameters()), lr=LEARNING_RATE, momentum=MOMENTUM)
loss_fn = nn.CrossEntropyLoss()

_, ohist = train_model(model_ft, dataloaders_dict, loss_fn, optimizer, NUM_EPOCHS)

# 下面是fine tuning的训练，并且不使用预训练的参数
model_scratch, _ = initialize_model(model_name, num_classes,
                    feature_extract=False, use_pretrained=False)
model_scratch = model_scratch.to(device)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, 
            model_scratch.parameters()), lr=LEARNING_RATE, momentum=MOMENTUM)
loss_fn = nn.CrossEntropyLoss()
_, scratch_hist = train_model(model_scratch, dataloaders_dict, loss_fn, optimizer, NUM_EPOCHS)


plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1, NUM_EPOCHS+1), ohist, label="Pretrained")
plt.plot(range(1, NUM_EPOCHS+1), scratch_hist, label="Scratch")
plt.ylim((0, 1.))
plt.xticks(np.arange(1, NUM_EPOCHS+1, 1.0))
plt.legend()
plt.show()

