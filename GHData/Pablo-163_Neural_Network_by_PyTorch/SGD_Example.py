import torch
import numpy as np
import matplotlib.pyplot as plt


def parabola(y):
    return (5 * y ** 2).sum()


k = torch.tensor([8., 9.], requires_grad=True)

optimizer = torch.optim.SGD([k], lr=0.06)


def do_grad_dis_step(func, var):
    func_res = func(var)
    func_res.backward()
    optimizer.step()
    optimizer.zero_grad()


# objects for visualistion
var_history = []
fn_history = []

for i in range(100):
    var_history.append(k.data.numpy().copy())
    fn_history.append(parabola(k).data.cpu().numpy().copy())
    do_grad_dis_step(parabola, k)

print(torch.less_equal(k, torch.tensor([0.01, 0.01])))

# Visualisation
def show_contours(objective,
                  x_lims=[-10.0, 10.0],
                  y_lims=[-10.0, 10.0],
                  x_ticks=100,
                  y_ticks=100):
    x_step = (x_lims[1] - x_lims[0]) / x_ticks
    y_step = (y_lims[1] - y_lims[0]) / y_ticks
    X, Y = np.mgrid[x_lims[0]:x_lims[1]:x_step, y_lims[0]:y_lims[1]:y_step]
    res = []
    for x_index in range(X.shape[0]):
        res.append([])
        for y_index in range(X.shape[1]):
            x_val = X[x_index, y_index]
            y_val = Y[x_index, y_index]
            res[-1].append(objective(np.array([[x_val, y_val]]).T))
    res = np.array(res)
    plt.figure(figsize=(7, 7))
    plt.contour(X, Y, res, 100)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')


show_contours(parabola)
plt.scatter(np.array(var_history)[:, 0], np.array(var_history)[:, 1], s=10, c='r')

plt.figure(figsize=(7, 7))
plt.plot(fn_history)
plt.xlabel('step')
plt.ylabel('function value')

plt.show()  # for some IDEs such PyCharm that line is needed to see graphs