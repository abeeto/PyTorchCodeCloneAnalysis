import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore') #ignore warnings

x = torch.linspace(-np.pi, np.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(1, 1001):
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print('No.{: 5d}, loss: {:.6f}'.format(t, loss.item()))
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 反向传播计算梯度
    optimizer.step() # 梯度下降法更新参数