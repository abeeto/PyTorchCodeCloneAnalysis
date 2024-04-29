import torch


def do_grad_dis_step(func, x, lr=0.01):
    func_x = func(x)
    func_x.backward()
    x.data -= lr * x.grad
    x.grad.zero_()  # PyTorch's grad is sumed default. So, make zero it


def parabola(y):
    return (5 * y ** 2).sum()


k = torch.tensor([8., 9.], requires_grad=True)
for i in range(100):
    do_grad_dis_step(parabola, k)

print(torch.less_equal(k, torch.tensor([0.01, 0.01])))
