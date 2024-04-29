from __future__ import print_function
import os
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import cv2
from torch.autograd import Variable
from resnet import ResNet18

#   权重文件，等于''时不使用预训练权重
model_path = r'/content/drive/MyDrive/github/PyTorch_CIFAR10-RESNET/pre_model/net_197.pth'
# 原始图像存储位置
image_dir = r'/content/drive/MyDrive/github/cifar10_dataset/clean_train/'
# image_dir = 'D:\\PycharmProjects\\AdversarialAttacksTest\\data\\'
# 攻击样本存储位置
save_dir = r'/content/drive/MyDrive/github/cifar10_dataset/clean_adv/'

# 获取计算设备 默认是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
  transforms.ToTensor()])

unloader = transforms.ToPILImage()

# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
  image = tensor.cpu().clone()
  image = image.squeeze(0)
  image = unloader(image)
  return image


def imshow(tensor, title=None):
  image = tensor.cpu().clone() # we clone the tensor to not do changes on it
  image = image.squeeze(0) # remove the fake batch dimension
  image = unloader(image)
  plt.imshow(image)
  if title is not None:
    plt.title(title)
  plt.pause(3.001) # pause a bit so that plots are updated


model = ResNet18().to(device)
#   加载权重
if model_path != '':
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_image_list(image_dir):
    image_list = []
    for file in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file)
        if file_path[-4:] == '.png':
            # print("file_path=", file_path)
            image_list.append(file_path)
    return image_list


def generate_aa_samples(image_path):
    orig = cv2.imread(image_path)[..., ::-1]
    img = orig.copy().astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = Variable(torch.from_numpy(img).to(device).float())
    tensor_adv = fast_gradient_method(model, img.data, 0.01, np.inf)
    adv_noise = tensor_to_PIL(tensor_adv)
    adv_image = orig + adv_noise
    cv2.imwrite(save_dir+"fgsm_"+os.path.basename(image_path), adv_image)


image_paths = get_image_list(image_dir)
i = 0
for i in range(len(image_paths)):
    image_path = image_paths[i]
    generate_aa_samples(image_path)
    print("generate {} image: {}".format(i, image_path))
