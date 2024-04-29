import torch
import torch.nn as nn

# function: f=2x
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return w * x


print(f'Prediction before training for f(5) {forward(5)}')

learning_rate = 0.01
n_iters = 50

# loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

# training
for epoch in range(n_iters):
    # forward pass
    y_pred = forward(x)
    # loss
    l = loss(y, y_pred)
    # backward pass
    l.backward()
    # update weight(s)
    optimizer.step()
    optimizer.zero_grad()
    print(f'epoch {epoch} : w = {w}, loss= {l}')
    print(f'Prediction after training for f(5) {forward(5)}')
