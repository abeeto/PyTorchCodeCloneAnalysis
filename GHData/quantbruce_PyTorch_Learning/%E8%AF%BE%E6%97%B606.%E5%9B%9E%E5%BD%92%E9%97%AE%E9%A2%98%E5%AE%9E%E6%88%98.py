import numpy as np
import pandas as pd


def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b))**2
    return totalError / float(len(points))


def step_gradient(b_cur, w_cur, points, learning_rate):
    b_grad = 0
    w_grad = 0
    N = float(len(points))
    for i in range(len(points)):  # 这里算的是批量梯度下降法，每更新一次new_b, new_w 需要遍历所有样本点
        x = points[i, 0]
        y = points[i, 1]
        b_grad += -(2/N) * (y - ((w_cur * x) + b_cur))
        w_grad += -(2/N) * x * (y - ((w_cur * x) + b_cur))
    new_b = b_cur - (learning_rate * b_grad)
    new_w = w_cur - (learning_rate * w_grad)
    return [new_b, new_w]


def gradient_descent_runner(points, start_b, start_w, learning_rate, num_iters):
    b = start_b
    w = start_w
    for i in range(num_iters):
        b, w = step_gradient(b, w, np.array(points), learning_rate) # 这里b, w每一轮值不断在更新
    return [b, w]


def run():
    points = np.genfromtxt(
        r'D:\geek growing\pytorch\深度学习与PyTorch入门实战教程_源码+课件\Deep-Learning-with-PyTorch-Tutorials\lesson04-简单回归案例实战\data.csv',
        delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iters = 1000
    print('Starting gradient descent at b = {0}, w = {1}, error = {2}'.format(initial_b, initial_w, compute_error_for_line_given_points(initial_b, initial_w, points)))
    print('Runing......')
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iters)
    print('After {0} iterations b = {1}, w = {2}, error = {3}'.format(num_iters, b, w, compute_error_for_line_given_points(b, w, points)))


if __name__ == '__main__':
    run()
