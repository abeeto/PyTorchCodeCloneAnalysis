from importlib.metadata import requires
import re
import numpy as np
import torch


class NerualNetwork_numpy:
    def __init__(self, input_num, output_num, nums_hidden_list) -> None:
        self.input_num = input_num
        self.output_num = output_num
        self.weights = []
        self.nums_hidden_list = nums_hidden_list

        in_num = input_num
        for i in range(len(nums_hidden_list)):
            W, b = self.init_weight(in_num, nums_hidden_list[i])
            self.weights.append((W, b))
            in_num = nums_hidden_list[i]
        self.weights.append(self.init_weight(in_num, output_num))

    def linear_layer(self, X, weight):
        y = X @ weight[0] + weight[1]
        return self.sigmoid(y)

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))
    
    def softmax(self, y):
        y_exp = np.exp(y)
        return y_exp / np.sum(y_exp, axis=1).reshape(-1, 1)

    def init_weight(self, in_num, out_num):
        W = np.random.randn(in_num, out_num)
        b = np.random.randn(out_num)
        return W, b
    
    def loss(self, y, y_hat):
        return np.mean(np.square(y.reshape(1, -1) - y_hat.reshape(1, -1)))

    def forward(self, X, y):
        assert X.shape[1] == self.input_num, "The input' dim is not match with model."
        hidden = X
        for i in range(len(self.nums_hidden_list)):
            output = self.linear_layer(hidden, self.weights[i])
            hidden = output
        y_hat = self.softmax(hidden @ self.weights[-1][0] + self.weights[-1][1])
        labels = np.argmax(y_hat, axis=1)
        loss = self.loss(y, labels)
        return y_hat, labels, loss


class NerualNetwork_torch:
    def __init__(self, input_num, output_num, nums_hidden_list) -> None:
        self.input_num = input_num
        self.output_num = output_num
        self.weights = []
        self.nums_hidden_list = nums_hidden_list # [16, 64, 32, 16]
        self.lr = 1e-1

        in_num = input_num
        for i in range(len(nums_hidden_list)):
            W, b = self.init_weight(in_num, nums_hidden_list[i])
            self.weights.append((W, b))
            in_num = nums_hidden_list[i]
        self.weights.append(self.init_weight(in_num, output_num))

    def linear_layer(self, X, weight):
        y = X @ weight[0] + weight[1]
        return self.sigmoid(y)

    def sigmoid(self, y):
        return 1 / (1 + torch.exp(-y))
    
    def softmax(self, y):
        y_exp = torch.exp(y)
        return y_exp / torch.sum(y_exp, axis=1).reshape(-1, 1)

    def init_weight(self, in_num, out_num):
        W = torch.randn(in_num, out_num, requires_grad=True)
        b = torch.randn(out_num, requires_grad=True)
        return W, b
    
    def loss(self, y, y_hat):
        return torch.mean(torch.square(y.reshape(1, -1) - y_hat.reshape(1, -1)))
    
    def update_grad(self):
        for W, b in self.weights:
            dW = W.grad; db = b.grad
            W = W.detach(); b = b.detach()
            W -= self.lr * dW
            b -= self.lr * db
            W.requires_grad_(True)
            b.requires_grad_(True)
    
    def train(self, X, y, epochs):
        assert X.shape[1] == self.input_num, "The input' dim is not match with model."

        for epoch in range(epochs):
            print("epoch %d: " % (epoch))
            hidden = X
            for i in range(len(self.nums_hidden_list)):
                output = self.linear_layer(hidden, self.weights[i])
                hidden = output
            y_hat = self.softmax(hidden @ self.weights[-1][0] + self.weights[-1][1])
            # index = torch.argmax(y_hat, dim=1, keepdim=True)
            # labels = torch.gather(y_hat,dim=1,index=index)
            # loss = self.loss(y, labels)
            loss = self.loss(y, y_hat)
            loss.backward()
            self.update_grad()
            print("loss: %.4f" % loss.item())
        # return y_hat, labels, loss
        return y_hat, loss


if __name__ == '__main__':
    use_numpy = True
    np.random.seed(23)
    X = np.random.rand(20, 10)
    y = np.random.randint(0, 2, size=(20))
    nums_hidden_list = [16, 64, 32, 16]
    
    if use_numpy:
        NN = NerualNetwork_numpy(10, 2, nums_hidden_list)
        y_hat, labels, loss = NN.forward(X, y)
        print('y: ', y.tolist())
        print('labels: ', labels)
        print('loss: ', loss.tolist())
    else:
        NN = NerualNetwork_torch(10, 1, nums_hidden_list)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        y_hat, loss = NN.train(X, y, 100)
        print('y: ', y.numpy().tolist())
        print('loss: ', loss.item())
