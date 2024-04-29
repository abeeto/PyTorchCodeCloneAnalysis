import torch

batch_size, D_in, hidden_size, D_out= 64, 1000, 100, 10
x = torch.randn(batch_size, D_in)
y = torch.randn(batch_size, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, D_out)
)

# This is a popular loss function. This is Mean Squared Error as our loss
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model
    # Input a Tensor of input dataand out Tensor of output data
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    # Zero the gradients before tehrunning the backward pass
    model.zero_grad()

    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad