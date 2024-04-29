import argparse
import time
from importlib import import_module
from collections import OrderedDict

import numpy as np
import torch

from 多个工具 import 构建数据集, 构建迭代器, 获得时间偏移量
from 训练和评估 import 训练模型

解析 = argparse.ArgumentParser(description='中文文本分类')
解析.add_argument('--模型', type=str, required=True, help='选择一个模型：形变双向编码器表示法库，文心大模型类')
复数参数 = 解析.parse_args()


def 缩减数据集(读取行数=512):
    """
    减少数据量，用于测试代码运行过程是否存在问题
    :return:
    """
    文件名列表 = ["测试集.txt", "训练集.txt", "验证集.txt"]
    for 文件名 in 文件名列表:
        行列表 = []
        with open("清华中文文本分类工具包/数据集/临时/" + 文件名, 'r', encoding="utf-8") as 文件:
            for i in range(读取行数):
                行列表.append(文件.readline())
            文件.close()
        with open("清华中文文本分类工具包/数据集/" + 文件名, 'w', encoding="utf-8") as 文件:
            文件.writelines(行列表)
            文件.close()

    exit()


def 修改预训练的模型():
    """
    为了适配中文命名，重新生成状态字典
    预训练中文模型： 该文件可以载入模型后使用模型的方法state_dict【模型.state_dict()】，使用for打印输出
    :return:
    """
    中文模型列表 = None
    with open('预训练中文模型', 'r', encoding='utf8') as 文件:
        中文模型列表 = 文件.readlines()
        文件.close()

    修改后的模型 = OrderedDict()
    预训练的模型 = torch.load("形变双向编码器表示法_预训练模型/火炬_模型.bin")

    for i, j in zip(预训练的模型, 中文模型列表):
        修改后的模型[j.strip('\n')] = 预训练的模型[i]

    # 元数据信息不匹配。。。。。。，也暂时不需要
    # 元数据 = getattr(预训练的模型, '_metadata', None)
    # 修改后的模型._metadata = 元数据
    torch.save(修改后的模型, '形变双向编码器表示法_预训练模型/火炬_中文_模型.bin')
    # print(预训练的模型)
    exit()


def 加载修改后的模型():
    """
    用于查看是否成功替换
    :return:
    """
    # 预训练的模型 = torch.load("形变双向编码器表示法_预训练模型/火炬_中文_模型.bin")
    预训练的模型 = torch.load("形变双向编码器表示法_预训练模型/火炬_模型.bin")
    # 元数据 = getattr(预训练的模型, '_metadata', None)
    # print(元数据)
    for i in 预训练的模型:
        print(预训练的模型[i])
        break
    exit()


if __name__ == '__main__':
    # 缩减数据集()
    # 修改预训练的模型()
    # 加载修改后的模型()

    # 需要添加参数 --模型 形变双向编码器表示法库
    数据集 = '清华中文文本分类工具包'

    模型名 = 复数参数.模型
    x = import_module('模型.' + 模型名)
    配置 = x.配置(数据集)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    开始时间 = time.time()
    print("正在载入数据......")
    训练用数据, 验证用数据, 测试用数据 = 构建数据集(配置)
    # 验证用数据 = 构建数据集(配置)

    训练用迭代器 = 构建迭代器(训练用数据, 配置)
    验证用迭代器 = 构建迭代器(验证用数据, 配置)
    测试用迭代器 = 构建迭代器(测试用数据, 配置)
    # next(训练用迭代器)
    时间偏移量 = 获得时间偏移量(开始时间)
    print("花费的时间：", 时间偏移量)

    模型 = x.模型(配置).to(配置.设备)  # 形变双向编码器表示法库文件的模型类
    # for i in 模型.state_dict():
    #     print(i)
    # exit()
    训练模型(配置, 模型, 训练用迭代器, 验证用迭代器, 测试用迭代器)
