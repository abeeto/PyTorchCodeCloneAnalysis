import torch

x = torch.tensor(
    [[1., 2., 3., 4.],
     [5., 6., 7., 8.],
     [9., 10., 11., 12.]], requires_grad=True
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

x = x.to(device)

function = 10 * (x ** 2).sum()

function.backward()  # backward pass to compute gradients
print(function.grad_fn)
print(function.grad_fn.next_functions[0][0])
print(function.grad_fn.next_functions[0][0].next_functions[0][0])
print(function.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])

print(x, '<- values of x')
print(x.grad, '<- gradient')

# refresh weights
x.data -= 0.0001 * x.grad
print(x, '<- values of x')

# to not cumulate gradients in tensor
x.grad.zero_()
print(x.grad, '<- gradient')

x = torch.tensor([8., 8.], requires_grad=True)


def function_parabola(variable):
    return 10 * (variable ** 2).sum()


def make_gradient_step(function, variable):
    func_res = function(variable)
    func_res.backward()
    variable.data -= 0.001 * variable.grad
    variable.grad.zero_()


for i in range(500):
    make_gradient_step(function_parabola, x)

print(x)

x = torch.tensor([8., 8.], requires_grad=True)
print(x)
optimizer = torch.optim.SGD([x], lr=0.001)


def make_gradient_step_optim(function, variable):
    func_res = function(variable)
    func_res.backward()
    optimizer.step()
    optimizer.zero_grad()


# for i in range(500):
#     make_gradient_step_optim(function_parabola, x)
#     # print(x)
# print(x)


def func(var):
    return var ** 2


x = torch.tensor([1.], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.09)
for i in range(50):
    make_gradient_step_optim(func, x)
print(x)
