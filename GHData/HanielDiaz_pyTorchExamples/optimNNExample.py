import torch

batch_size, D_in, hidden_size, D_out= 64, 1000, 100, 10

x = torch.randn(batch_size, D_in)
y = torch.randn(batch_size, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, D_out)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()