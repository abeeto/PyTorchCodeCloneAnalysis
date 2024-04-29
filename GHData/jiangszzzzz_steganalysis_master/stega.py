# This is a sample Python script.
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip
)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from glob import glob
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from sklearn import metrics
import torch.nn.functional as F
from dataset import Alaska2Dataset

import model
from YeNet import YeNet

## 随机数 seed 生成相同随机数
seed = 42
print(f'setting everything to seed {seed}')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

## 创建train 和 test
data_dir = '/steganalysis/dataset'
sample_size = 50000
val_size = int(sample_size * 0.25)
train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ['Cover_jpg/', 'JMiPOD/', 'JUNIWARD/', 'UERD/']  # label 1 2 3

for label, folder in enumerate(folder_names):
    # python的glob模块可以对文件夹下所有文件进行遍历，并保存为一个list列表
    train_filenames = glob(f"{data_dir}/{folder}/*.jpg")[:sample_size]
    np.random.shuffle(train_filenames)
    # train
    train_fn.extend(train_filenames[val_size:])
    train_labels.extend(np.zeros(len(train_filenames[val_size:], )) + label)
    # val
    val_fn.extend(train_filenames[:val_size])
    val_labels.extend(np.zeros(len(train_filenames[:val_size], )) + label)

assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"

train_df = pd.DataFrame({'ImageFileName': train_fn, 'Label': train_labels}, columns=['ImageFileName', 'Label'])
train_df['Label'] = train_df['Label'].astype(int)
val_df = pd.DataFrame({'ImageFileName': val_fn, 'Label': val_labels}, columns=['ImageFileName', 'Label'])
val_df['Label'] = val_df['Label'].astype(int)

print(train_df)
train_df.Label.hist()

img_size = 512
AUGMENTATIONS_TRAIN = Compose([
    Resize(img_size, img_size, p=1),
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    JpegCompression(quality_lower=75, quality_upper=100, p=0.5),
    ToFloat(max_value=255),
    ToTensorV2()
], p=1)
AUGMENTATIONS_TEST = Compose([
    Resize(img_size, img_size, p=1),
    ToFloat(max_value=255),
    ToTensorV2()
], p=1)

batch_size = 32
num_workers = 2

# 构建数据集s
train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
valid_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size * 2, num_workers=num_workers,
                                           shuffle=False)

device = 'cuda'
# 选择模型
# model = model.Efficientnet4()
# model = model.Srnet()
# model = model.SSrnet()
model = YeNet()

model.to(device)

# 多卡并行
device_ids = [0, 1, 2]  # 选中其中3块

# torch规定：必须把参数放置在nn.DataParallel中参数device_ids[0]上，在这里device_ids=[1,2]，因此我们需要 device=torch.device("cuda:1" )
model = nn.DataParallel(model, device_ids=device_ids)

# 固定ssrnet 前两层
# 冻结
# for name, param in model.named_parameters():
#     if 'layer0' in name:
#         param.requires_grad = False

# 优化器传入参数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        # pdb.set_trace()

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization


criterion = torch.nn.CrossEntropyLoss()
num_epochs = 30
train_loss, val_loss = [], []

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    model.train()
    running_loss = 0
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for im, labels in tk0:
        inputs = im["image"].to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tk0.set_postfix(loss=(loss.item()))

    epoch_loss = running_loss / (len(train_loader) / batch_size)
    train_loss.append(epoch_loss)
    print('Training Loss: {:.8f}'.format(epoch_loss))

    tk1 = tqdm(valid_loader, total=int(len(valid_loader)))
    model.eval()
    running_loss = 0
    y, preds = [], []
    with torch.no_grad():
        for (im, labels) in tk1:
            inputs = im["image"].to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            y.extend(labels.cpu().numpy().astype(int))
            preds.extend(F.softmax(outputs, 1).cpu().numpy())
            running_loss += loss.item()
            tk1.set_postfix(loss=(loss.item()))

        epoch_loss = running_loss / (len(valid_loader) / batch_size)
        val_loss.append(epoch_loss)
        preds = np.array(preds)
        # 多分类到二分类
        labels = preds.argmax(1)
        acc = (labels == y).mean() * 100
        new_preds = np.zeros((len(preds),))
        temp = preds[labels != 0, 1:]
        new_preds[labels != 0] = temp.sum(1)
        new_preds[labels == 0] = preds[labels == 0, 0]
        y = np.array(y)
        y[y != 0] = 1
        auc_score = alaska_weighted_auc(y, new_preds)
        print(f'Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}')

# 保存模型路径
    torch.save(model.state_dict(), f"./Yenet/epoch_{epoch + 4}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth")

plt.figure(figsize=(15, 7))
plt.plot(train_loss, c='r')
plt.plot(val_loss, c='b')
plt.legend(['train_loss', 'val_loss'])

plt.savefig("Yenetloss.png")
plt.title('Loss Plot')
