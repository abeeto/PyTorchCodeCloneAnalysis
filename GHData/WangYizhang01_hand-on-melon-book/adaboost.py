# 手写实现Adaboost算法
# 解决《统计学习方法》中例8.1，基学习器选择广义符号函数（具体见原书）
# by WangYizhang
import numpy as np


X = np.arange(10)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

class BaseLearner:
    '''
    基学习器，使用广义符号函数，对阈值v，G(x) = 1 if x <= v; G(x) = -1 if x > v.
    Args:
        X、y: 输入的训练数据及标签, ndarray
        X_weights: 数据集的权重, ndarray
    '''
    def __init__(self, X, y, X_weights) -> None:
        self.X = X
        self.y = y
        self.X_weights = X_weights
        self.v = np.random.randint(1) # 阈值
    
    def error(self, v):
        y_hat = 1 * (self.X <= v) + (-1) * (self.X > v)
        error = np.sum(self.X_weights * (y_hat != y))
        return error
    
    def fit(self):
        min_error = 11; min_v = -1
        for v in range(np.min(self.X) - 1, np.max(self.X) + 1):
            error = self.error(v)
            if error < min_error:
                min_error = error
                min_v = v
        self.v = min_v
    
    def predict(self):
        y_hat = 1 * (self.X <= self.v) + (-1) * (self.X > self.v)
        return y_hat


class Adaboost:
    '''
    简易版AdaBoost算法
    Args:
        X、y: 输入的训练数据及标签, ndarray
        m: 基学习器的个数, int
    '''
    def __init__(self, X, y, m) -> None:
        self.X = X
        self.y = y
        self.m = m # 基学习器个数
        self.alpha = []

    def fit(self):
        alpha = []
        D = np.ones(self.X.size) / self.X.size
        G = []
        V = []
        for i in range(self.m):
            # 使用权值分布D的训练数据集学习，得到基本分类器Gi
            Gi = BaseLearner(self.X, self.y, D)
            Gi.fit()
            # 计算Gi在训练数据集上的分类误差率
            e = Gi.error(Gi.v)
            # 计算Gi的系数alpha_i
            alpha_i = 0.5 * np.log((1 - e) / e)
            alpha.append(alpha_i)
            # 更新训练数据集的权值分布
            update_D = D * np.exp(-1 * alpha_i * self.y * Gi.predict())
            update_D = update_D / np.sum(update_D)
            D = update_D
            G.append(Gi)
            V.append(Gi.v)
        # 构建基本分类器的线性组合
        f = np.zeros(self.y.size)
        for i in range(len(G)):
            f += alpha[i] * G[i].predict()
        
        Gx = 1 * (f >= 0) + (-1) * (f < 0)
        return Gx, alpha



if __name__ == '__main__':
    ad = Adaboost(X, y, 3)
    Gx, alpha = ad.fit()
    print("Gx: ", Gx)
    print('alpha: ', alpha)
