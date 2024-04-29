from turtle import forward
import numpy as np
import torch


class LogisticRegression_torch():
    def __init__(self, X, y) -> None:
        self.w = torch.randn(X.shape[1], requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lr = 1e-1

    def forward(self, epochs):
        for i in range(epochs):
            y_hat = torch.sigmoid(self.X @ self.w + self.b)
            loss = torch.mean(torch.square(self.y - y_hat))
            loss.backward()

            dl_dw = self.w.grad; dl_db = self.b.grad
            if i % 100 == 0: print('epoch: ', i, ' loss: ', loss.item())

            self.w = self.w.detach(); self.b = self.b.detach()
            self.w -= dl_dw * self.lr
            self.b -= dl_db * self.lr
            self.w.requires_grad_(True)
            self.b.requires_grad_(True)
        print("最终结果: ", self.w, self.b)


if __name__ == '__main__':
    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s
        
    np.random.seed(123)
    X = np.random.rand(100, 5)
    noise = np.random.rand(X.shape[0])
    w = np.random.randint(1, 10, size=5)
    print('true_w: ', w)
    # y = X @ w + 5 + noise
    y = sigmoid(X @ w + 5 + noise)

    LR = LogisticRegression_torch(X, y)
    LR.forward(1000)
