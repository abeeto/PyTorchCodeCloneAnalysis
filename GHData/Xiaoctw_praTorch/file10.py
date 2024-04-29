from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


def show_landmarks(image, landmarks):
    plt.imshow(image)
    # 画点
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)


class FaceLandmarksDataSet(Dataset):
    """
    定义了一个数据集类，数据集中包括图片和配置信息，需要的时候读取图片
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """

        :param csv_file: csv文件路径
        :param root_dir: 所有图像的目录
        :param transform: 可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        """
        返回一个实例，实例包括图片和对应的内容信息
        直接通过索引的方式获取
        :param idx:
        :return:
        """
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            # 如果存在转换就进行转换
            sample = self.transform(sample)
        return sample


face_dataset = FaceLandmarksDataSet(csv_file='faces/face_landmarks.csv',
                                    root_dir='faces/')
fig = plt.figure()
for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)
    # 这里所有数字都不能超过10
    # 这里三个数字表示行，列和索引值
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title("Sample #{}".format(i))
    ax.axis("off")
    show_landmarks(**sample)
    if i == 3:
        plt.show()
        break


# 接下来进行数据变换，因为神经网络假设图片的尺寸相同，进行预处理
class Rescale():
    """
    对图片进行缩放，缩放到给定大小
     output_size（tuple或int）：所需的输出大小。
     如果是元组，则输出为
         与output_size匹配。
         如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
    """

    def __init__(self, output_size):
        # 如果不是一个整形或者元组，那么就返回false
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]  # image的shape为三维，获得前两维
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # 在skimage包中，对图片进行操作
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]  # 坐标进行变化
        return {'image': img, 'landmarks': landmarks}


class RandomCrop():
    """
    对给定图片进行随机裁剪
    """

    def __init__(self, output_size):
        """
        :param output_size:
        所需的输出大小。 如果是int，方形裁剪.
        """
        assert isinstance(output_size, (tuple, int))
        if isinstance(output_size, int):
            # 进行方形裁剪
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top:top + new_h, left:left + new_w]
        landmarks -= [left, top]
        return {'image': image, 'landmarks': landmarks}


class ToTensor():
    """
    将样本中ndarrays的数据转化成Tensors
    numpy包的图片是: H * W * C
    torch包的图片是: C * H * W
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))  # 交换轴
        return {
            'image': torch.from_numpy(image),
            'landmarks': torch.from_numpy(landmarks)
        }


scale = Rescale(256)  # 缩放
crop = RandomCrop(128)  # 随机裁剪
# 定义一个组合变换
composed = transforms.Compose([Rescale(256), RandomCrop(224)])
fig = plt.figure()
sample = face_dataset[64]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)
plt.show()
