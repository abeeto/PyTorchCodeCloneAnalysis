import numpy as np

"""
Building a neural network using numpy from scratch
Including ReLU activation function, cross entropy cost function, and gradient descent backpropagation
The implementation is tested to be correct
"""
class Matmul:
    def __init__(self):
        self.mem = {}

    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem = {'x': x, 'W': W}
        return h

    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        x = self.mem['x']
        W = self.mem['W']

        grad_x = np.matmul(grad_y, W.T)  # shape(N, b)
        grad_W = np.matmul(x.T, grad_y)
        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}

    def forward(self, x):
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']
        return (x > 0).astype(np.float32) * grad_y


class Softmax:
    '''
    softmax over last dimention
    '''

    def __init__(self):
        self.epsilon = 1.e-8
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        # fixme This softmax may overflow when x is large
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / (partition + self.epsilon)
        #print(x_exp[:3, :3], out[:3, :3])

        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))  # (N, c, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj)  # (N, 1, c)
        tmp = np.squeeze(tmp, axis=1)
        tmp = -tmp + grad_y * s
        return tmp


class Log:
    '''
    softmax over last dimention
    '''

    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x + self.epsilon)

        self.mem['x'] = x
        return out

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']

        return 1. / (x + 1e-12) * grad_y


class Model_NP:
    def __init__(self, num_inputs, num_outputs, num_hiddens = 100, lr = 1.e-5, lambda1 = 0.001):
        self.W1 = np.random.normal(size=[num_inputs + 1, num_hiddens])
        self.W2 = np.random.normal(size=[num_hiddens, num_outputs])

        self.mul_h1 = Matmul()
        self.mul_h2 = Matmul()
        self.relu = Relu()
        self.softmax = Softmax()
        self.log = Log()

        self.lr = lr
        self.lambda1 = lambda1
        self.num_hiddens = num_hiddens

    def forward(self, x):

        # ==========
        # todo '''the forward propagating process of MLP which has a structure of FFN --> RELU --> FFN --> Softmax'''
        # ==========

        # The input x has size (60000, 28, 28)
        # The first layer has 28 * 28 neurons and a bias
        # The second layer has 100 neurons
        # The output layer has 10 neurons
        # change the shape of x to (60000, 28 * 28)
        n_sample, h, w = x.shape
        x = x.reshape(n_sample, h * w)
        # add one bias to x and make it (60000, 28 * 28 + 1)
        x = np.insert(x, 0, 1, axis=1)
        x = self.relu.forward(self.mul_h1.forward(x, self.W1)) # x is (60000, 100)
        # softmax + log is the cross entropy loss
        x = self.softmax.forward(self.mul_h2.forward(x, self.W2)) # x is (60000, 10)
        self.h2_log = self.log.forward(x)
        return self.h2_log


    def backward(self, label):

        # ==========
        # todo '''back propagation process'''
        # ==========
        # according to the calculation of derivative of cross entropy loss function, the minus sign is important
        output_grad = self.softmax.backward(-self.log.backward(label)) # output_grad is (60000, 10)
        grad_x, grad_W = self.mul_h2.backward(output_grad) # grad_x is (60000, 100), grad_W is (100, 10)
        self.h2_grad = grad_W # doesn't need to divide sample number
        grad_x, grad_W = self.mul_h1.backward(self.relu.backward(grad_x)) # grad_x is (60000, 785), grad_W is (785, 100)
        self.h1_grad = grad_W

    def update(self):

        # ==========
        # todo '''update the parameters of MLP'''
        # ==========
        self.W1 -= self.lr * (self.h1_grad + self.lambda1 * self.W1)
        self.W2 -= self.lr * (self.h2_grad + self.lambda1 * self.W2)


if __name__ == '__main__':
    model = Model_NP()