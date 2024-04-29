import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('Logs_3')
device = torch.device('cuda')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
).to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    writer.add_scalar('Loss', loss.item(), i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



