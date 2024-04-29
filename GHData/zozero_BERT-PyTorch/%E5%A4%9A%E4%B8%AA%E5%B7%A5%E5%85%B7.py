import time

import torch
from tqdm import tqdm
from datetime import timedelta

填充, 分类 = '[PAD]', '[CLS]'


def 构建数据集(配置):
    def 载入数据(路径, 句子长度=32):
        内容列表 = []
        with open(路径, 'r', encoding='UTF-8') as 文件:
            for 行 in tqdm(文件):
                一行 = 行.strip()
                if not 一行:
                    continue
                内容, 标签 = 一行.split('\t')
                # 由于我使用的是中文，以字为单位
                字符列表 = 配置.分词器.分词(内容)
                字符列表 = [分类] + 字符列表
                字符列表长度 = len(字符列表)
                掩码 = []
                字符标记列表 = 配置.分词器.把字符转换到标记(字符列表)

                if 句子长度:
                    if len(字符列表) < 句子长度:
                        掩码 = [1] * len(字符标记列表) + [0] * (句子长度 - len(字符列表))
                        字符标记列表 += ([0] * (句子长度 - len(字符列表)))
                    else:
                        掩码 = [1] * 句子长度
                        字符标记列表 = 字符标记列表[:句子长度]
                        字符列表长度 = 句子长度
                内容列表.append((字符标记列表, int(标签), 字符列表长度, 掩码))
        return 内容列表

    训练用数据集 = 载入数据(配置.训练_路径, 配置.句子长度)
    验证用数据集 = 载入数据(配置.验证_路径, 配置.句子长度)
    测试用数据集 = 载入数据(配置.测试_路径, 配置.句子长度)
    return 训练用数据集, 验证用数据集, 测试用数据集
    # return 验证用数据集


class 数据集迭代器:
    def __init__(self, 数据集, 每批数量, 设备):
        self.每批数量 = 每批数量
        self.数据集 = 数据集
        self.批数 = len(数据集) // 每批数量
        self.是否有余数 = False  # 数据集%每批数量是否有余数
        if len(数据集) % self.批数 != 0:
            self.是否有余数 = True
        self.索引 = 0
        self.设备 = 设备

    def _转成张量(self, 数据):
        x = torch.LongTensor([_[0] for _ in 数据]).to(self.设备)
        y = torch.LongTensor([_[1] for _ in 数据]).to(self.设备)

        数据_数量 = torch.LongTensor([_[2] for _ in 数据]).to(self.设备)
        掩码 = torch.LongTensor([_[3] for _ in 数据]).to(self.设备)

        return (x, 数据_数量, 掩码), y

    def __next__(self):
        if self.是否有余数 and self.索引 == self.批数:
            数据集 = self.数据集[self.索引 * self.每批数量:len(self.数据集)]
            self.索引 += 1
            数据集 = self._转成张量(数据集)
            return 数据集
        elif self.索引 > self.批数:
            self.索引 = 0
            raise StopIteration
        else:
            数据集 = self.数据集[self.索引 * self.每批数量:(self.索引 + 1) * self.每批数量]
            self.索引 += 1
            数据集 = self._转成张量(数据集)
            return 数据集

    def __iter__(self):
        return self

    def __len__(self):
        if self.是否有余数:
            return self.批数 + 1
        else:
            return self.批数


def 构建迭代器(数据集, 配置):
    迭代器 = 数据集迭代器(数据集, 配置.每批数量, 配置.设备)
    return 迭代器


def 获得时间偏移量(开始时间):
    结束时间 = time.time()
    时间偏移量 = 结束时间 - 开始时间
    return timedelta(seconds=int(round(时间偏移量)))
