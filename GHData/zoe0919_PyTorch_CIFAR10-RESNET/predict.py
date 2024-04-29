import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import time
import numpy as np
import json

from custom_dataset.dataset import MyDataSet

#   Cifar-10的标签：('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_path = r"C:\Users\baoti\Desktop\PyTorch_CIFAR10-RESNET\model\net_197.pth"

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

#   测试集目录
test_root = r"C:\Users\baoti\Desktop\PyTorch_CIFAR10-RESNET\images\clean_random"
#   标签
labels = r"C:\Users\baoti\Desktop\PyTorch_CIFAR10-RESNET\images\clean_train\clean_label.txt"

#   超参数设置
#   遍历数据集次数
EPOCH = 1
#   批处理尺寸(batch_size)
BATCH_SIZE = 128
#   使用的内核数
nw = 0

#   测试集归一化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#   模型定义-ResNet
net = ResNet18().to(device)

#   加载权重
if model_path != '':
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = net.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
else:
    assert 0, "Need pre-train weight!"

def read_txt_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    #   分割成图片名 + label
    tokens = [l.strip().split(' ') for l in lines]
    #   返回字典
    return dict(((name, label) for name, label in tokens))

def read_test_data(data_dir: str):
    #   文件夹不存在则报错
    assert os.path.exists(data_dir), "dataset root: {} does not exist.".format(data_dir)

    #   生成类别名称以及对应的数字索引
    #   返回的字典格式为：{'0.jpg':'0', '1.jpg':'0', ...}
    #   注意拼接前不要加\
    class_indices = read_txt_labels(os.path.join(data_dir, r"clean_label.txt"))
    #   将val和key反向
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    #   存储测试集的所有图片路径
    test_images_path = []
    #   存储测试集图片对应索引信息
    test_images_label = []
    #   支持的文件后缀类型
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for train_file in os.listdir(data_dir):
        #   判断是不是支持的文件格式
        if os.path.splitext(train_file)[-1] in supported:
            #   获取label，对应字典value
            label = class_indices[train_file]
            #   获取图片路径
            fname = os.path.join(data_dir, train_file)

            #   添加测试集图片路径
            test_images_path.append(fname)
            #   添加测试集图片标签
            test_images_label.append(int(label))
    print("{} images for test.".format(len(test_images_path)))
    return test_images_path, test_images_label

#   验证
if __name__ == "__main__":
    test_images_path, test_images_label = read_test_data(test_root)
    test_set = MyDataSet(images_path=test_images_path, images_class=test_images_label, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for epoch in range(EPOCH):
        #   开始测试
        #   不更新梯度
        with torch.no_grad():
            total = 0
            for i, data in enumerate(test_loader, 0):
                net.eval()
                images, label, filename = data
                images = images.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)

                #   追加写，开始下一次完整预测之前，需要手动删除prediction_clean_random.txt
                f4 = open("prediction_clean_random.txt", 'a+')
                for i in range(BATCH_SIZE):
                    f4.write("{} {}\n".format(os.path.basename(filename[i]), predicted[i]))
            f4.close()

        end = time.time()

    print("Training Finished, TotalEPOCH=%d" % EPOCH)
