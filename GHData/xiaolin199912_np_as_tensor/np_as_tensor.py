"""
作者:林泽毅
建这个开源库的起源呢,是因为在做onnx推理的时候,需要将原来的tensor转换成numpy.array
问题是Tensor和Numpy的矩阵排布逻辑不同
包括Tensor推理经常会进行Transform,比如ToTensor,Normalize等
就想做一些等价转换的函数。
"""
import numpy as np
from torchvision import transforms
from PIL import Image
import torch


def NTo_Tensor(array):
    """
    :param array: opencv/PIL读取的numpy矩阵
    :return:返回一个形如Tensor的numpy矩阵

    Example:
    Inputs:array.shape = (512,512,3)
    Outputs:output.shape = (3,512,512)
    """
    output = array.transpose((2, 0, 1))
    return output


def NNormalize(array, mean, std, dtype=np.float32):
    """
    :param array: opencv/PIL读取的numpy矩阵
           mean: 归一化均值,np.array格式
           std:  归一化标准差,np.array格式
           dtype：输出的numpy数据格式,一般onnx需要float32
    :return:numpy矩阵

    Example:
    Inputs:array为opencv/PIL读取的一张图片
           mean=np.array([0.5,0.5,0.5])
           std=np.array([0.5,0.5,0.5])
           dtype=np.float32
    Outputs:output为归一化后的numpy矩阵
    """
    im = array / 255.0
    im = np.divide(np.subtract(im, mean), std)
    output = np.asarray(im, dtype=dtype)

    return output

def NUnsqueeze(array, axis=0):
    """
    :param array: opencv/PIL读取的numpy矩阵
           axis：要增加的维度
    :return:numpy矩阵

    Example:
    Inputs:array为opencv/PIL读取的一张图片,array.shape为[512,512,3]
           axis=0
    Outputs:output为array在第0维增加一个维度,shape转为[1,512,512,3]
    """
    if axis == 0:
        output = array[None, :, :, :]
    elif axis == 1:
        output = array[:, None, :, :]
    elif axis == 2:
        output = array[:, :, None, :]
    else:
        output = array[:, :, :, None]

    return output

if __name__ == "__main__":
    input_path = "./test.jpg"
    image_np = Image.open(input_path)
    image_np = np.asarray(image_np.copy())
    print("[0]Numpy_shape:", image_np.shape)

    #=========1.ToTensor==============#
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image_np.copy())
    to_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image_tensor = to_normalize(image_tensor)
    # print(image_tensor)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    print("[1]Tensor_shape:", image_tensor.shape)

    # =========2.Normalize==============#

    image_np = NNormalize(image_np, mean=np.array([0.5,0.5,0.5]), std=np.array([0.5,0.5,0.5]))
    image_np = NTo_Tensor(image_np)
    # print(image_np)
    image_np = NUnsqueeze(image_np, 0)

    print("[2]Numpy_shape:", image_np.shape)






