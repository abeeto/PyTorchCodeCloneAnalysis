import torch as t
import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('Logs')

device = t.device('cuda')

N, D_in, H, D_out = 64, 1000, 100, 10

x = t.randn(N, D_in, device=device)
y = t.randn(N, D_out, device=device)

w1 = t.randn(D_in, H, device=device, requires_grad=True)
w2 = t.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6

for i in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    writer.add_scalar('Loss', loss.item(), i)

    loss.backward()

    with t.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()
writer.close()
