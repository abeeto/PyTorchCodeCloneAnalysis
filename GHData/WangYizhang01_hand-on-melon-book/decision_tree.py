# ID3
# by WangYizhang
import math

import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self) -> None:
        pass

    def load_data(self, filename):
        '''
        导入数据
        Args:
            filename: csv文件存放地址
        Returns:
            data_list: 从csv中读取的数据集,每一行为一个样本,最后一列为标签,前面为特征
            label_list: 列名组成的列表
        '''
        dataset = pd.read_csv(filename)
        # 标签名
        label_list = list(dataset.columns.values)
        # 数据集（最后一列是标签y）
        data_list = dataset.values
        # 删除编号列（第1列）,仅用于 watermelon.csv
        data_list = np.delete(data_list, 0, axis=1)
        label_list = label_list[1:]
        return data_list, label_list
    
    def calcShannonEnt(self, data_list):
        '''
        计算给定数据集的熵
        Args:
            data_list: 数据集
        Returns:
            ent: 熵
        '''
        l = len(data_list); labs = [exp[-1] for exp in data_list]
        classDict = {}
        for lab in labs:
            if lab not in classDict.keys():
                classDict[lab] = 1
            else:
                classDict[lab] += 1
        ent = - np.sum([val/l * math.log(val/l, 2) for lab, val in classDict.items()]).item()
        return ent

    def splitDatasetbyFeature(self, data_list, feat_index, feat_val):
        '''
        从数据集中找出满足给定的特征-值对关系的数据子集
        Args:
            data_list: 数据集
            feat_index: 特征序号
            feat_val: 特征值
        Returns:
            sub_data_list: 满足给定的特征-值对关系的数据子集
        '''
        sub_data_list = []
        for sample in data_list:
            if sample[feat_index] == feat_val:
                sub_data_list.append(list(sample[:feat_index]) + list(sample[feat_index+1:]))
        return sub_data_list

    def calcInfoGain(self, data_list, feat_index):
        '''
        计算数据集关于某个特征的信息增益
        Args:
            data_list: 数据集
            feat_index: 特征序号
        Returns:
            feat_info_gain: 该特征的信息增益
        '''
        l = len(data_list); feat_info_gain = self.calcShannonEnt(data_list)
        feat_values = set([sample[feat_index] for sample in data_list])
        for feat_val in feat_values:
            sub_data_list = self.splitDatasetbyFeature(data_list, feat_index, feat_val)
            feat_info_gain -= len(sub_data_list) / l * self.calcShannonEnt(sub_data_list)
        return feat_info_gain

    def chooseBestFeat(self, data_list):
        '''
        返回数据集信息增益最大的特征序号
        Args:
            data_list: 数据集
        Returns:
            best_feat_index: 信息增益最大的特征序号
        '''
        feat_num = len(data_list[0]) - 1
        best_feat_index = -1; best_info_gain = 0.
        for i in range(feat_num):
            feat_info_gain = self.calcInfoGain(data_list, i)
            if feat_info_gain > best_info_gain: 
                best_feat_index = i
                best_info_gain = feat_info_gain
        return best_feat_index

    def majorityCnt(self, class_list):
        '''
        返回类别列表中出现最多的标签
        Args:
            class_list: 类别列表
        Returns:
            label: 出现最多的标签
        '''
        counter = dict(zip(*np.unique(class_list, return_counts=True)))
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        label = counter[0][0]
        return label
    
    def createTree(self, data_list, label_list):
        """
        构造决策树
        @ param data_list: 数据集
        @ param labels: 标签集
        @ return myTree: 决策树
        """
        classList = [sample[-1] for sample in data_list]
        # 若当前数据集中样本全属于同一类别，返回该类别
        if classList.count(classList[0]) == len(classList):
            return classList[0]

        # 若特征集合为空，即已遍历完所有特征值时，返回出现次数最多的类别
        if (len(data_list[0]) == 1):
            return self.majorityCnt(classList)
        
        # 获取最佳划分属性
        best_feat_index = self.chooseBestFeat(data_list)
        bestFeatLabel = label_list[best_feat_index]
        myTree = {bestFeatLabel:{}}
        # 清空labels[bestFeat]
        del label_list[best_feat_index]
        featValues = [example[best_feat_index] for example in data_list]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = label_list[:]
            # 递归调用创建决策树
            myTree[bestFeatLabel][value] = self.createTree(self.splitDatasetbyFeature(data_list, best_feat_index, value), subLabels)
        return myTree


if __name__ =='__main__':
    filename = './datasets/watermelon.csv'
    # filename = './datasets/isFish.csv'
    DT = DecisionTree()
    data_list, label_list = DT.load_data(filename)
    tree = DT.createTree(data_list, label_list)
    print(tree)
