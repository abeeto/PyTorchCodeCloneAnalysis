
import torch
import time

class MatrixExponentEstimator(torch.nn.Module):
    def __init__(self):
        super(MatrixExponentEstimator, self).__init__()
        self.d0 = torch.nn.Linear(4, 32)
        torch.nn.init.xavier_normal_(self.d0.weight)
        self.t0 = torch.nn.Tanh()
        self.d1 = torch.nn.Linear(32, 16)
        torch.nn.init.xavier_normal_(self.d1.weight)
        self.t1 = torch.nn.Tanh()
        self.d2 = torch.nn.Linear(16, 4)
        torch.nn.init.xavier_normal_(self.d2.weight)

    def forward(self, x):
        x = self.t0(self.d0(x))
        x = self.t1(self.d1(x))
        return self.d2(x)

def f(x):
    return torch.matrix_exp(x.reshape((2,2))).reshape((4,))

def apply_matrix_exponential(x):
    return torch.stack([f(x_i) for x_i in torch.unbind(x)])

def train():
    epochs = 10000
    trainx = torch.randn(10000, 2*2)
    trainy = apply_matrix_exponential(trainx)
    testx = torch.randn(10000, 2*2)
    testy = apply_matrix_exponential(testx)

    model = MatrixExponentEstimator()
    adam = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.MSELoss()
    print('Initial Train Loss: {:.4f}'.format(loss_fn(model(trainx), trainy)))
    print('Initial Test Loss: {:.4f}'.format(loss_fn(model(testx), testy)))
    for _ in range(3):
        t_start = time.time()
        for _ in range(epochs):
            adam.zero_grad()
            loss_fn(model(trainx), trainy).backward()
            adam.step()
        print('Took: {:.2f} seconds'.format(time.time() - t_start))
        print('Train Loss: {:.4f}'.format(loss_fn(model(trainx), trainy)))
        print('Test Loss: {:.4f}'.format(loss_fn(model(testx), testy)))
                
if __name__ == '__main__':
    train()


