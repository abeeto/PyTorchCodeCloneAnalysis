import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, filename='./datasets/isFish.csv') -> None:
        self.data_dict, self.dataset_size = self.load_data(filename)

    def load_data(self, filename):
        '''
        导入数据
        Args:
            filename: 数据源
        Returns:
            data_dict: 存储数据集的字典，key是标签值，value是标签值对应的样本集(ndarray)
            dataset_size: 数据集中的样本数
        '''
        dataset = pd.read_csv(filename)
        dataset_size = dataset.values.shape[0]
        label_list = list(dataset.columns.values)
        labels = np.unique(dataset[label_list[-1]].values.tolist())

        data_dict = {}
        for label in labels:
            data_dict[label] = dataset[dataset[label_list[-1]] == label].values

        return data_dict, dataset_size

    def predict(self, X):
        '''
        给定新样本X(单个)，返回朴素贝叶斯预测的类别
        Args:
            X: 新样本
        Returns:
            label: X的预测标签
        '''
        label = -1; max_val = -1; D = self.dataset_size
        for lab in self.data_dict.keys():
            sub_datasets = self.data_dict[lab]
            D_c = sub_datasets.shape[0]
            cur_val = D_c / D
            for i in range(X.size):
                count_i = np.sum(sub_datasets[:, i] == X[i])
                cur_val *= (count_i / D_c)
            if cur_val > max_val:
                max_val = cur_val; label = lab
        return label
    
    def predict_batch(self, batch_X):
        '''
        给定新样本X(batch)，返回朴素贝叶斯预测的类别
        Args:
            X: 新样本
        Returns:
            label: X的预测标签
        '''
        label_batch = []
        for X in batch_X:
            label_batch.append(self.predict(X).item())
        return label_batch


if __name__ == '__main__':
    NB = NaiveBayes()
    batch_X = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
    print(NB.predict_batch(batch_X))
