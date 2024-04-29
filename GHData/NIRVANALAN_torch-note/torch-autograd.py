import torch
import time

N, D_in, H, D_out = 64, 1000, 100, 10
device = torch.device('cuda')
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

lr = 1e-6
T = 500
t1 = time.time()

for t in range(5000):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 += -lr * w1.grad
        w2 += -lr * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
print('time', time.time() - t1)
'''
gpu
4999 2.3030877116525517e-07
time 6.548532962799072
cpu
4999 2.9209024887677515e-07
time 3.8776447772979736
'''