import os
import numpy as np
import skimage.io as io
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):  # 继承Dataset
    def __init__(self, root_dir, augment=None, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.augment = augment
        self.transform = transform  # 变换
        imagepath = []
        for dir in os.listdir(self.root_dir):  # 目录里的所有文件
            temps = os.listdir(os.path.join(self.root_dir, dir))
            for i in range(len(temps)):
                temps[i] = dir + '/' + temps[i]
            imagepath += temps
        self.images = imagepath

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = io.imread(img_path)  # 读取该图片 OpenCV, PIL, sci-kit image(skimage)
        label = img_path.split('/')[2]  # 根据该图片的路径名获取该图片的label
        sample = {'image': img, 'label': label}  # 根据图片和标签创建字典

        if self.transform:
            sample['image'] = np.asarray(self.transform(Image.fromarray(sample['image'])))  # 对样本进行变换

        if self.augment:
            sample['image'] = self.augment(sample['image'])
        return sample  # 返回该样本

datasets = MyDataset(root_dir="./allset/",
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.RandomCrop(224),
                               transforms.ToTensor()
                           ]))

dataloader = DataLoader(datasets, batch_size=1, shuffle=True)

# 使用数据集加载器的例子: 获取一个batch
# num_epoches = 100
# for epoch in range(num_epoches):
#     for img, label in dataloader:
#         #这里是训练的代码
#         print("training...")

# 使用数据集加载器的例子: 另一种获取batch的方法
for i_batch,batch_data in enumerate(dataloader):
        print(i_batch)#打印batch编号
        print(batch_data['image'].size())#打印该batch里面图片的大小，相当于 img
        print(batch_data['label'])#打印该batch里面图片的标签, 相当于 label



# 训练集和测试集的分割
# train_size = int(len(datasets) * 0.7)
# test_size = len(datasets) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])
