import torch
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0], requires_grad=True)


# our model forward pass
def forward(x):
    return x * w

# Loss function
def loss(y_pred, y):
    return (y_pred - y) ** 2


# Before training
print("predict (before training)", 4, forward(4).item())

mse_list = []
# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)  # 1) Forward pass
        l = loss(y_pred, y_val)  # 2) Compute loss
        l.backward()  # 3) Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.item()

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    mse_list.append(l.data[0].item())
    print(f"Epoch: {epoch} | LOss: {l.item()}")
    print()

# After training
print("predict (after training)", 4, forward(4).item())

plt.plot(mse_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()