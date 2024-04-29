#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ data_processor.py
 Author @ huangjunheng
 Create date @ 2018-06-17 16:06:27
 Description @ 
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data


def file2array(filename):
    """
    file to array
    :param filename: 
    :return: 
    """
    ret_array = []
    fr = open(filename)
    for line in fr:
        line = line.rstrip('\n')
        ret_array.append(line)

    return ret_array


def mnist2text():
    """
    mnist data to text
    :return: 
    """
    ret_text_array = []
    batch_size = 100
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    def tensor2lines(images, labels):
        """

        :return: 
        """
        batch_lines = []
        np_images = images.numpy()
        np_labels = labels.numpy()

        num_classes = len(set(np_labels))

        for sample, label in zip(np_images, np_labels):
            line = ''
            one_hot_label = [0] * num_classes
            for s in sample:
                line += '#'.join([str(i) for i in s]) + '\t'

            one_hot_label[int(label)] = 1
            line = line.rstrip('\t') + '&' + '\t'.join([str(i) for i in one_hot_label])
            print line

            batch_lines.append(line)

        return batch_lines

    for i, (images, labels) in enumerate(train_loader): # test_loader

        images = images.reshape(-1, 28, 28)
        ret_text_array.extend(tensor2lines(images, labels))

    # print ret_text_array
    return ret_text_array


def cal_model_para(filename):
    """
    根据数据计算模型的参数
    1. 单个输入特征的维度: input_size
    2. label的维度，几分类就几个维度: num_class
    :param filename: 
    :return: 
    """
    fr = open(filename)
    for i, line in enumerate(fr):
        line = line.rstrip('\n')
        data_split = line.split('&')
        feature_data_list = data_split[0].split('\t')

        if i == 0:
            input_size = len(feature_data_list[0].split('#'))
            num_class = len(data_split[1].split('\t'))

    print 'According to "%s", input_size is set to %d, num_class is set to %d.' \
          % (filename, input_size, num_class)
    return input_size, num_class


class DataProcessor(object):
    """
    数据处理
    """
    def __init__(self, filename, batch_size, shuffle=False):
        """
        init
        :param filename: 
        :param batch_size: 
        :param shuffle: 
        """
        self.filename = filename
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load(self):
        """
        file to data_loader
        :return: 
        """
        line_array = file2array(self.filename)
        x_tensor, y_tensor = self._lines2tensor_xy(line_array)
        torch_dataset = Data.TensorDataset(x_tensor, y_tensor)

        # 把 dataset 放入 DataLoader

        data_loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        return data_loader

    def _lines2tensor_xy(self, line_list=['1#3\t2#5\t3#7&1\t0', '3#3\t5#5\t7#7&0\t1']):
        """
        convert line to tensor_x and tensor_y
        :param line: 
        :return: 
        """
        batch_feature_list = []
        batch_label_list = []
        for line in line_list:
            split_datas = line.split('&')
            x_line = split_datas[0]
            y_line = split_datas[1]

            batch_label_list.append(y_line.split('\t'))

            sequence_features = x_line.split('\t')
            sequence_list = []
            for sequence_feature in sequence_features:
                sequence_list.append(sequence_feature.split('#'))

            batch_feature_list.append(sequence_list)

        np_batch_feature = np.array(batch_feature_list, dtype=np.float32)
        x_tensor = torch.from_numpy(np_batch_feature)

        y_tensor = torch.from_numpy(np.array(batch_label_list, dtype=np.int64))

        return x_tensor, y_tensor

    def main(self):
        """
        test func
        :return: 
        """
        data_loader = self.load()

        for x, y in data_loader:
            print x, y


if __name__ == '__main__':
    filename = 'data/training_test_data/test_data.txt'
    batch_size = 100

    loader = DataProcessor(filename, batch_size)
    loader.main()

    # mnist2text()









