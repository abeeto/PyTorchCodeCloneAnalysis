import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Hyper-parameter
input_size = 784
n_classes = 10
n_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='data/',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

test_dataset = torchvision.datasets.MNIST(root='data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Logistic regression model
model = nn.Linear(input_size, n_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+1)%100 == 0:
            print('Epochs [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1 , n_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum()

    print("# ====================================== #")
    print('#            Accuracy: {} %              #'.format(100*correct/total))
    print("# ====================================== #")

# Save the model checkpoint
torch.save(model.state_dict(), 'logistic_regression_model.ckpt')

