import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# Convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Visualize sample images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(12, 3))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(str(labels[idx].item()))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define the layers
        self.hidden_1 = nn.Linear(28*28, 512)
        self.hidden_2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Apply the layers
        x = x.view(-1, 28*28)
        x = F.relu(self.hidden_1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_2(x))
        x = self.dropout(x)
        x = self.output(x) # output is logits
        return x

# initialzie the NN
model = Net()
print(model)

# specify the loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1

for epoch in range(epochs):
    train_loss = 0  # monitor training loss

    for images, labels in train_loader:
        optimizer.zero_grad()  # clear the gradients
        outputs = model(images)  # forward pass
        loss = criterion(outputs, labels)  # calcualte the loss
        loss.backward()  # backward pass
        optimizer.step()  # parameter update
        train_loss += loss.item()

    train_loss = train_loss/len(train_loader)

    print('Epoch {} \tTraining loss: {:.6f}'.format(epoch+1, train_loss))

# Test the model
test_loss = 0
class_correct = list(0 for i in range(10))
class_total = list(0 for i in range(10))

model.eval()  # prep for model evaluation

for images, labels in test_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)
    test_loss += loss.item()
    _, pred = torch.max(outputs, 1)
    correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
    for i in range(len(images)):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# Calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

