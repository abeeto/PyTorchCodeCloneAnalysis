import torch

dtype = torch.float
device= torch.device("cpu")
# device = torch.device("cuda:0")

batch_size, D_in, hidden_size, D_out = 64, 1000, 100, 10

# These are just random starting data so we can see our loss
x = torch.randn(batch_size, D_in, device=device, dtype=dtype)
y = torch.randn(batch_size, D_out, device=device, dtype=dtype)

# These are random weight but they will be updated as the algortihm learns
w1 = torch.randn(D_in, hidden_size, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(hidden_size, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Here we multiply by all the weights and don't save any values because pytorch
    # is going to do mm for us
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # This just calculates the loss based off of the distance from the answer to the
    # prediction
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # This call will compute the radient of loss with respect to all Tensors that have
    # requires_grad=True. Calling w1.grad or w2.grad will give us the gradient of both of
    # those weights.
    loss.backward()

    with torch.no_grad():
        # Updates the weights based off the learning rate and the gradients
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # This makes all the gradients zero
        w1.grad.zero_()
        w2.grad.zero_()


