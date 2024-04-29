import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

# Initialize parameters
W1 = torch.randn(784, 500) / np.sqrt(784)
W1.requires_grad_()
b1 = torch.zeros(500, requires_grad=True)

W2 = torch.randn(500, 10) / np.sqrt(500)
W2.requires_grad_()
b2 = torch.zeros(10, requires_grad=True)

# Optimizer
optimizer = torch.optim.SGD([W1, W2, b1, b2], lr=1.0)

# Iterate through train set minibatchs
print("Training the model")
for images, labels in tqdm(train_loader):
    # Zero out the gradients
    optimizer.zero_grad()

    # Forward pass
    x = images.view(-1, 28 * 28)
    a1 = torch.matmul(x, W1) + b1
    y1 = F.relu(a1)
    y2 = torch.matmul(y1, W2) + b2
    cross_entropy = F.cross_entropy(y2, labels)
    # Backward pass
    cross_entropy.backward()
    optimizer.step()

## Testing
correct = 0
total = len(mnist_test)

print("Testing the model")
with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, 28 * 28)
        a1 = torch.matmul(x, W1) + b1
        y1 = F.relu(a1)
        y2 = torch.matmul(y1, W2) + b2

        predictions = torch.argmax(y2, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct / total))
