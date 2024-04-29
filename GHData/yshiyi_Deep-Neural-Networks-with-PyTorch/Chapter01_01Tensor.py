import numpy as np
import torch
import matplotlib.pyplot as plt
import sys


# Plot vectors, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector,
# "color": color of the vector on diagram}]

def plotVec(vectors):
    ax = plt.axes()
    # For loop to draw the vectors
    for vec in vectors:
        # vec["vector"] contains two values, we must use * to unpack them.
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"],
                 head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.show()


# %%
##########################################################
# 1D Tensor
##########################################################
a = torch.tensor([-1, 1, 2])
# print(a.type(), a.dtype)  # type: tensor.LongTensor, dtype: tensor.int64
print(a.size())
a_2d = a.view(-1, 1)  # Convert to a 3*1 column 2d tensor
print(a, a_2d)
# af = a.type(torch.FloatTensor)
# # Note: mean(), sin(), these special functions can deal with float datatype.
# print(a, torch.mean(a, dtype=torch.float), a.max())
# print(a[0])
# print(a[0].item())
# print(af, af.mean(), af.max(), torch.sin(af))
#
# b = torch.linspace(-5, 5, steps=3)
# print(b)
# x = torch.linspace(0, 2*np.pi, 100)
# y = torch.sin(x)
# plt.plot(x, y)  # plt.plot(x.numpy(), y.numpy()) convert to a numpy array
# plt.show()
#
# ----------------------------------- #
# # Convert a numpy array to a tensor
# ----------------------------------- #
# numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
# new_tensor = torch.from_numpy(numpy_array)
# back_to_numpy = new_tensor.numpy()
# print(numpy_array, new_tensor, back_to_numpy)
# # back_to_numpy and new_tensor are both pointing to numpy_array.
# # As a result, if we modify numpy_array, both back_to_numpy and new_tensor
# # will be changed.
# # "=": equal sign is a shallow copy.
# numpy_array[0] = 6.0
# print(numpy_array, new_tensor, back_to_numpy)
#
# # np.c_: Add along second axis
# print(np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])])  # (3,1) & (3,1) => (3,2)
# print(np.c_[np.array([[1, 2, 3]]), np.array([[4, 5, 6]])])  # (1,3) & (1,3) => (1,6)
# sys.exit(0)
#

# %%
# ----------------------------------- #
# Slicing
# ----------------------------------- #

# Create tensors
# u = torch.tensor([1, 0])
# v = torch.tensor([0, 1])
# w = u + v
# plotVec([
#     {"vector": u.numpy(), "name": 'u', "color": 'r'},
#     {"vector": v.numpy(), "name": 'v', "color": 'b'},
#     {"vector": w.numpy(), "name": 'w', "color": 'g'}
# ])
#
# x = torch.arange(-2.0, 2.0, 0.5).view(-1, 1)
# y = torch.zeros(x.size())  # a column vector
# y2 = torch.zeros(x.shape[0])  # a row vector
# print(x, x[:, 0], x.size(), x.shape[0])
# print(y, y2)
# y[(x > 1.0)] = 1.0
# y2[(x > 1.0)[:, 0]] = 1.0  # use [:, 0] to make x become a row vector
# print(y, y2, y[:, 0])
# print((x > -1.0))
# print((x[:, 0] > -1.0))
# print((x > -1.0) * (x < 1.0))
# a = (x > -1.0) * (x < 1.0)
# b = (x[:, 0] > -1.0) & (x[:, 0] < 1.0)
# print(a.shape, a.size())
# print(b.shape, b.size())
# print(a.shape, a[:, 0], y.shape)
# y[(x > -1.0) * (x < 1.0)] = 1.0
# print(y)
# # sys.exit(0)
#
# a = torch.tensor([1, 2, 3])
# a2 = torch.tensor([[1], [2], [3]])
# print(a, a2)
# print(a.shape, a2.shape)

# %%
# Slicing in Python is different from Matlab
# Here, b is a 2*3 2D tensor. If we select the 3rd column
# using [:, 2], it gives us tensor([3, 6]). Its size is 2.
# It is not a 2*1 column 2D tensor.
# b = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(b, b[:, 2], b[1], b[1, :])
# sys.exit(0)
#
# l = np.array([1, 2, 3, 4])
# for i, j, k in zip(l, l[1:], l[2:]):
#     print(i, j, k)  # 1 2 3 \n 2 3 4
# sys.exit(0)
#
# print(np.zeros((3, 1)).shape)
#
#
# y = np.zeros(3)
# y = torch.from_numpy(y)
# print(y.shape)
# print(y.view(-1).shape)
# print(y.view(1, -1), y.view(1, -1).shape)
# print(y.view(-1, 1))

# %%
# L = np.array([-1, 1, 2])
# M = np.array([[1, 1, 1], [2, 2, 2]])
# N = np.matrix([[1, 1, 1], [2, 2, 2]])
# print('Starts from here:')
# print(M * L, L * M)  # Element-wise multiplication [[-1,1,2], [-2,2,4]]
# # if we use torch, then torch.size() and torch.shape are the same
# print(M.shape, N.shape)  # (2, 3)
# print(M.size, N.size)  # 6, 6
# print(M[:, 0], N[:, 0])  # [1, 2], [[1]; [2]]
# print(M[:, 0].shape, N[:, 0].shape)  # (2,) (2, 1)

# x = np.matrix([1, 2]).T
# y = np.matrix(np.random.rand(3, 2))
# print('Starts from here:')
# print(y * x)  # matrix multiplication
# print(y)
# print(np.multiply(y[:, 0], y[:, 1]))  # element wise

# xt = torch.tensor([[1, 2]])
# yt = torch.tensor([[1, 2], [3, 4], [5, 6]])
# print('Starts from here:')
# print(xt * yt)  # element wise
# print(torch.mul(xt, yt))  # element wise
# print(torch.mm(yt, torch.t(xt)))  # matrix multiplication
# print(torch.mm(yt, xt.t()))
# print(yt.transpose(1, 0))


# %%
##########################################################
# 2D Tensor
##########################################################
# # Convert 2D List to 2D Tensor
# twoD_list = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
# twoD_tensor = torch.tensor(twoD_list)
# print(twoD_tensor.shape, twoD_tensor.size(),
#       twoD_tensor.numel(), twoD_tensor.ndim)
#
# ----------------------------------- #
# # Convert tensor to numpy array;
# # Convert numpy array to tensor
# ----------------------------------- #
# twoD_numpy = twoD_tensor.numpy()
# print("Tensor -> Numpy Array:")
# print("The numpy array after converting: ", twoD_numpy)
# print("Type after converting: ", twoD_numpy.dtype)
# print("================================================")
# new_twoD_tensor = torch.from_numpy(twoD_numpy)
# print("Numpy Array -> Tensor:")
# print("The tensor after converting:", new_twoD_tensor)
# print("Type after converting: ", new_twoD_tensor.dtype)
#
# ----------------------------------- #
# # Slicing
# ----------------------------------- #
# tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
# # [1:3]: extract 2nd and 3rd rows,
# # [0]: 1st row from previous step, [0]: 1st element from previous step
# print(tensor_example[1:3][0][0])  # tensor(21)
# print(twoD_tensor[1, 0], twoD_tensor[1][0])  # tensor(21)
#
# # General operators: +, -, * and / are all elementwise operations
# t1 = torch.tensor([[1., 2.], [3., 4.]])
# t2 = torch.tensor([[2., 2.], [2., 2.]])
# print(t1/t2)
#
# # Matrix multiplication
# t3 = torch.mm(t1, t2)
# print(t3)
# %%
# # Inner product
# t11 = torch.tensor([1.0, 2.0, 3.0, 4.0])
# t22 = torch.tensor([2.0, 2.0, 2.0, 2.0])
# t4 = torch.dot(t11, t22)
# print(t4)

# %%
##########################################################
# Derivative of tensors
##########################################################

# -------------------------#
# EX 01:
# -------------------------#
# requires_grad=True: can let us retrieve the derivative/gradient
x = torch.tensor(2.0, requires_grad=True)
print("The tensor x: ", x)
y = x ** 2
print("The result of y = x^2: ", y)
y.backward()
# x.grad: derivative of y w.r.t. x
print("The derivative at x = 2: ", x.grad, x.grad.item(), x.grad.data)


# # -------------------------#
# # EX 02:
# # -------------------------#
# # Partial derivative
# u = torch.tensor(1.0, requires_grad=True)
# v = torch.tensor(2.0, requires_grad=True)
# f = u * v + u ** 2
# print("The result of v * u + u^2: ", f)
# f.backward()
# print("The partial derivative with respect to u: ", u.grad)
# print("The partial derivative with respect to u: ", v.grad)


# # -------------------------#
# # EX 03:
# # -------------------------#
# # detach(): detach the variable from the calculation of derivative
# x = torch.tensor([1., 2.], requires_grad=True)
# y = x**2
# z = x**3
# R = y + z
# r = R.sum()
# print(x, y, z, R, r)
# r.backward()
# print(x.grad)  # r_dot = 2x + 3x^2
#
# # z = x.detach() ** 3
# # r = (y+z).sum()
# # print(y, z, r)
# # r.backward()
# # print(x.grad)  # r_dot = 2x
# print(x.detach())
# print(R.detach())
# plt.plot(x.detach(), R.detach(), label='function')
# plt.plot(x.detach(), x.grad, label='derivative')
# plt.xlabel('x')
# plt.legend()
# plt.show()

# # -------------------------#
# # EX 04:
# # -------------------------#
# # # Calculate the derivative with respect to a function with multiple values
# # # as follows. You use the sum trick to produce a scalar valued function
# # # and then take the gradient:
# x = torch.linspace(-10, 10, 10, requires_grad=True)
# Y = x ** 2
# y = Y.sum()
# # y = torch.sum(x ** 2)
# y.backward()  # Works like calculating partial derivative of y w.r.t. 10 different x
# # v = torch.ones(10, requires_grad=True)
# # Y.backward(v)  # 10 Y associated with 10 x, calculate Jacobian then times v
# #
# # # detach() excludes further tracking of operations in the graph,
# # # and therefore the subgraph will not record operations.
# # # This allows us to then convert the tensor to a numpy array.
# # plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
# # plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
# plt.plot(x.detach(), Y.detach(), label='function')
# plt.plot(x.detach(), x.grad, label='derivative')
# plt.xlabel('x')
# plt.legend()
# plt.show()


# # -------------------------#
# # EX 05:
# # -------------------------#
# x = torch.tensor([-1., 0, 1.], requires_grad=True)
# y = x * 2
# while y.data.norm() < 1000:
#     y = y * 2
# print(y)
# v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# y.backward(v)
# print(x.grad)
