import torch
import numpy as np
import time

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

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.2*num_train))  # 20% will be validation set
train_idx, val_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation sets
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_data, batch_size=32, sampler=val_sampler)
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

# Use GPU?
cuda = True
if cuda:
    model.cuda()
else:
    model.cpu()

# number of epochs
epochs = 20

# initialize tracker for minimum validation loss
val_loss_min = np.Inf

start = time.time()
for epoch in range(epochs):
    train_loss = 0  # monitor training loss
    val_loss = 0  # monitor vaalidation loss

    # train batch loop
    for images, labels in train_loader:
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()  # clear the gradients
        outputs = model(images)  # forward pass
        loss = criterion(outputs, labels)  # calcualte the loss
        loss.backward()  # backward pass
        optimizer.step()  # parameter update
        train_loss += loss.item()

    # validation batch loop
    model.eval()
    for images, labels in val_loader:
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    model.train()

    # Calculate average loss for epoch
    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)

    print('Epoch {} \tTraining loss: {:.6f} \tValidation loss: {:.6f}'.format(epoch+1, train_loss, val_loss))

    if val_loss <= val_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}. Saving model'. format(
            val_loss_min, val_loss
        ))
        torch.save(model.state_dict(), 'best_model.pth')
        val_loss_min = val_loss

print("Elapsed time: {}".format(time.time()-start))

# Load the best model parameters and test it
model.load_state_dict(torch.load('best_model.pth'))

# Test the model
test_loss = 0
class_correct = list(0 for i in range(10))
class_total = list(0 for i in range(10))

model.cpu()
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

