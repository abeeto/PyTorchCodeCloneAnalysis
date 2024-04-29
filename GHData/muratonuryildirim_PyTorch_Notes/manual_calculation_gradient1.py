import torch

# function: f=2x
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return w * x


# loss : MSE
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()


print(f'Prediction before training for f(5) {forward(5)}')

# training
learning_rate = 0.01
n_iters = 50

for epoch in range(n_iters):
    # forward pass
    y_pred = forward(x)
    # loss
    l = loss(y, y_pred)
    # backward pass
    l.backward()
    # update weight(s)
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()

    print(f'epoch {epoch} : w = {w}, loss= {l}')
    print(f'Prediction after training for f(5) {forward(5)}')


