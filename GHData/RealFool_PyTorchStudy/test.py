import numpy as np
class PCA():
    def __init__(self, n_components):
        self.n_components = n_components  # 降维的维度数

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]  # 数组的列数
        # 求协方差矩阵
        X = X - X.mean(axis=0)  # 计算每一列的均值
        self.covariance = np.dot(X.T, X) / X.shape[0]  # np.dot 矩阵乘法  shape[0]数组的行数
        print("协方差矩阵为：\n", self.covariance)
        # 求协方差矩阵的特征向量和特征值
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)  # 求特征值和特征向量的接口
        print("协方差矩阵的特征值是：\n", eig_vals)
        print("协方差矩阵的特征向量是：\n", eig_vectors)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
        print("特征值从大到小排序的序列号是：\n", idx)
        self.n_components_ = eig_vectors[:, idx[:self.n_components]]  # 降维矩阵
        print("降维矩阵为：\n", self.n_components_)
        return np.dot(X, self.n_components_)
pca = PCA(n_components=2)
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
newX = pca.fit_transform(X)
print("降维后的数据是：\n", newX)  # 输出降维后的数据