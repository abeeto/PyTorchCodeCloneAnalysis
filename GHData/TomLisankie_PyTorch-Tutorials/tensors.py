import torch 

dtype = torch.float
device = torch.device("cpu")

batch_size, input_dim, hidden_dim, output_dim = 64, 100, 1000, 10

# Random input and output data
x = torch.randn(batch_size, input_dim, device = device, dtype = dtype)
y = torch.randn(batch_size, output_dim, device = device, dtype = dtype)

# Initialize with random weights
w1 = torch.randn(input_dim, hidden_dim, device = device, dtype = dtype)
w2 = torch.randn(hidden_dim, output_dim, device = device, dtype = dtype)

learning_rate = 1e-6
epochs = 500
for t in range(epochs):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t + 1, loss)

    # Backprop to compute gradients of w1 and w2 with respect to the loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2