import numpy as np
N, D_in, H, D_out = 64, 1000, 100, 10 # 64个输入，输入是1000维

# 随机创建一些训练数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)   # 1000维到100维
w2 = np.random.randn(H, D_out)  # 100维到10维

learning_rate = 1e-6
for it in range(500):
    # Forward pass
    h = x.dot(w1)   # N * H
    h_relu = np.maximum(h, 0)   # N * H
    y_pred = h_relu.dot(w2)     # N * D_out
    
    # compute loss
    loss = np.square(y_pred - y).sum()
    print(it, loss)
    
    # Backward pass
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)       # 导数/梯度，y_pred为变量  N * D_out
    grad_w2 = h_relu.T.dot(grad_y_pred)    # w2为变量    H*N . N*D_out
    grad_h_relu = grad_y_pred.dot(w2.T)    # h_relu 为变量    N*D_out . D_out * H
    grad_h = grad_h_relu.copy()            # 先复制下，为下一个操作准备的
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)      # D_in * N . N * H
    
    # update weights of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2