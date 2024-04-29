import torch
import LeNet5
import os
from PIL import Image
import numpy as np


net = LeNet5.LeNet5()
if os.path.exists('./MNIST_model.ph'):
    net.load_state_dict(torch.load('./MNIST_model.ph'))
    net.eval()
    print('模型加载成功！')
else:
    print('暂时没有模型文件，请先训练模型后再测试！')
    exit()

# 要识别数字的路径
img_path = './mnist_png/mnist_png/testing/7/1276.png'
img = Image.open(img_path).convert('L')
img = np.array(img)
img = img.reshape(1, 28, 28)
if img.max() > 1:
    img = img / 255
img = torch.from_numpy(img)
img = torch.unsqueeze(img, dim=0)
img = img.float()
pred = net(img)
pred = torch.argmax(pred, dim=1).item()
print('识别的数字为：' + str(pred))

