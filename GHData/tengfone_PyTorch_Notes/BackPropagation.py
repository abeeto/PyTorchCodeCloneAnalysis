import torch
import pdb

x_data = [1,2,3]
y_data = [2,4,6]

w = torch.tensor([1.0], requires_grad= True)

# Forward Pass
def forward(x):
    return x * w

# Loss Function
def loss(y_pred, y_val):
    return (y_pred - y_val)**2

#Before training
print("Prediction before training: ", 4 , forward(4).data[0])

# Training loop
for i in range(10):
    for x_val, y_val in zip (x_data, y_data):
        y_pred = forward(x_val)
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print(f"Epoch: {i} | Loss: {l.item()}")

# After training
print("Prediction after training: ", 4, forward(4).item())
