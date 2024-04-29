import numpy as np
import torch
N, D_in, H, D_out = 64, 1000, 100, 10 # 64个输入，输入是1000维

# 说明：基于1.1版本，使用部分torch接口进行梯度的自动计算

# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H, requires_grad=True)   # 1000维到100维
w2 = torch.randn(H, D_out, requires_grad=True)  # 100维到10维

learning_rate = 1e-6
for it in range(500):
    # Forward pass
    # h = x.mm(w1)   # N * H
    # h_relu = h.clamp(min=0)   # N * H
    # y_pred = h_relu.mm(w2)     # N * D_out
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
    # compute loss
    loss = (y_pred - y).pow(2).sum()
    print(it, loss.item())
    
    # Backward pass
    # compute the gradient

    loss.backward()
    with torch.no_grad():
	    # update weights of w1 and w2
	    w1 -= learning_rate * w1.grad
	    w2 -= learning_rate * w2.grad
	    w1.grad.zero_()
	    w2.grad.zero_()
