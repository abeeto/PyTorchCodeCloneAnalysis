import numpy as np
from autograd import Tensor, Parameter, Module
from autograd.optim import SGD
x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, +3, -2], dtype=np.float))  # (3,)

# @矩阵乘法
# （100，3）*（3，1）
y_data = x_data @ coef + 5


class Module(Module):
    def __init__(self) -> None:
        self.w = Parameter(3)
        self.b = Parameter()

    def predict(self, in_puts: Tensor):
        return in_puts @ self.w + self.b


# w = Tensor(np.random.randn(3), requires_grad=True)
# b = Tensor(np.random.randn(), requires_grad=True)
# w = Parameter(3)  # tensor(3,),requires_grad =True,random values
# b = Parameter()

optimizer = SGD(lr=0.001)
# learning_rate = 0.001
batch_size = 32
module = Module()
for epoch in range(100):
    epoch_loss = 0.0
    for start in range(0, 100, batch_size):

        end = start + batch_size

        module.zero_grad()
        inputs = x_data[start:end]

        predicted = module.predict(inputs)
        actual = y_data[start:end]
        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss.data

        # module.w -= module.w.grad * learning_rate
        # module.b -= module.b.grad * learning_rate
        optimizer.step(module)
    print(epoch, epoch_loss)
