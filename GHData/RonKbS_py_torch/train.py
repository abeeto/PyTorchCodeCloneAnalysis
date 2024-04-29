import torch
from torch import nn
from torchvision import datasets, transforms

# Define transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load training data
trainset = datasets.MNIST(
    "~/.pytorch/MNINST_data/",
    download=True,
    train=True,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


'''
# build feed-forward network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Define the loss
criterion = nn.CrossEntropyLoss()

# Get unser data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and labels
loss = criterion(logits, labels)

print(loss)
'''


# ~build model with log-softmax output using nn.LogSoftmax
    # or F.log_softmax. Actual probabilities then gotten with
    # torch.exp(output). For this case, negative-log-likelihood-loss
    # nn.NLLLoss is more appropriate

# model returnning log-softmax as output
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    # removing the layer below returns a 2.3299 loss, much closer to that shown
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
'''
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

# forward-pass to get log-probabilities
logps = model(images)

# calculate loss with logps and labels
loss = criterion(logps, labels)

# print(loss)

print("Before backward pass : \n", model[0].weight.grad)

loss.backward()

print("After backward pass: \n", model[0].weight.grad)
'''

from torch import optim
# optimizers require parameters to optimize and a learning-rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

'''
print("Initial weights - ", model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# clear grads coz they are accumulated as multiple backward passes
    # with the same parameters are done
optimizer.zero_grad()


# Forward pass then backward pass, then update of the weights
output = model(images)
loss = criterion(output, labels)
loss.backward()
print("Gradient - ", model[0].weight.grad)

optimizer.step()
print("Updated weights - ", model[0].weight)
'''

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss : {running_loss/len(trainloader)}")

images, labels = next(iter(trainloader))
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# ouptut of network is log-probability, so take exponential for probabilities
ps = torch.exp(logps)

import helper
helper.view_classify(img.view(1, 28, 28), ps)
