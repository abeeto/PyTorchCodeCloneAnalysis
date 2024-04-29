# 手写实现线性回归（梯度下降法）
# by wangyizhang
import numpy as np
import matplotlib.pyplot as plt
import torch

# 生成数据
np.random.seed(123)
X = np.random.rand(100, 20) * 10
noise = np.random.rand(X.shape[0])
w = np.random.randint(1, 10, size=20)
print('true_w: ', w)
y = X @ w + 5 + noise
# plt.scatter(X, y)
# plt.show()

class LinearRegression_numpy():
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.w = np.random.rand()
        self.b = np.random.rand()
        self.lr = 0.01

    def fit(self, epochs):
        for i in range(epochs):
            y_hat = self.X * self.w + self.b
            loss = np.mean(np.square(self.y - y_hat))
            if i % 100 == 0: print('epoch: ', i, 'loss: ', loss)
            dl_dw = np.mean((y_hat - self.y) * X * 2)
            self.w -= dl_dw * self.lr
            dl_db = np.mean((y_hat - self.y) * 2)
            self.b -= dl_db * self.lr
        print('最终结果是', self.w, self.b)

class MultipleLinearRegression_numpy():
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        # 将wx+b转换成w‘x的形式，即在X第一列插入1
        self.X = np.insert(self.X, 0, 1, axis=1)
        self.w = np.random.rand(self.X.shape[1])
        self.lr = 0.00001

    def fit(self, epochs):
        for i in range(epochs):
            y_hat = self.X @ self.w # + self.b
            loss = np.mean(np.square(self.y - y_hat))
            if i % 100 == 0: print('epoch: ', i, 'loss: ', loss)
            dl_dw = self.X.T @ (y_hat - y)
            self.w -= dl_dw * self.lr

        print('最终结果是', self.w)

class LinearRegression_torch():
    def __init__(self, X, y) -> None:
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        w = np.random.rand()
        self.w = torch.tensor(w, requires_grad=True)
        b = np.random.rand()
        self.b = torch.tensor(b, requires_grad=True)
        self.lr = 0.01
    
    def fit(self, epochs):
        for i in range(epochs):
            y_hat = self.X * self.w + self.b
            loss = torch.mean(torch.square(self.y - y_hat))
            if i % 100 == 0: print('epoch: ', i, 'loss: ', loss.item())
            loss.backward()
            dl_dw = self.w.grad
            dl_db = self.b.grad
            self.w = self.w.detach(); self.b = self.b.detach()
            self.w -= dl_dw * self.lr
            self.b -= dl_db * self.lr
            self.w.requires_grad = True; self.b.requires_grad = True
        print('最终结果是', self.w, self.b)

class MultipleLinearRegression_torch():
    def __init__(self, X, y) -> None:
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        w = np.random.rand(self.X.shape[1])
        b = np.random.rand()
        self.w = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        self.b = torch.tensor(b, dtype=torch.float32,requires_grad=True)
        self.lr = 0.001
    
    def fit(self, epochs):
        for i in range(epochs):
            y_hat = self.X @ self.w + self.b
            loss = torch.mean(torch.square(self.y - y_hat))
            if i % 100 == 0: print('epoch: ', i, 'loss: ', loss.item())
            loss.backward()
            dl_dw = self.w.grad
            dl_db = self.b.grad
            self.w = self.w.detach(); self.b = self.b.detach()
            self.w -= dl_dw * self.lr
            self.b -= dl_db * self.lr
            self.w.requires_grad = True; self.b.requires_grad = True
        print('最终结果是', self.w, self.b)


if __name__ == "__main__":
    # 一元线性回归
    # LR = LinearRegression_numpy(X, y)
    # LR = LinearRegression_torch(X, y)
    # 多元线性回归
    # LR = MultipleLinearRegression_numpy(X, y)
    LR = MultipleLinearRegression_torch(X, y)
    LR.fit(1000)
