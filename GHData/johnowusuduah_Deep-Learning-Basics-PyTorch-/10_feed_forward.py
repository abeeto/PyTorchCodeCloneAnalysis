import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# OUTLINE
#************************************************************************************************************
# 1. MNIST
# 2. DATA LOADER, TRANSFORMATION
# 3. MULTILAYER NEURAL NET, ACTIVATION FUNCTION
# 4. LOSS AND OPTIMIZER
# 5. TRAINING LOOP (USING BATCH TRAINING)
# 6. MODEL EVALUATION
# 7. GPU SUPPORT


# i. HYPERPARAMETERS
#************************************************************************************************************
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# ii. DATA
#************************************************************************************************************
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

example = iter(train_loader)
samples, labels = example.next()
print(samples.shape, labels.shape) # 100 - no. samples in batch, 1 - no. of channels(greyscale), 28,28 - pixel dimension


# plot six examples from sample
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap="gray")
plt.show()


# iii. FEED FORWARD NEURAL NETWORK
#************************************************************************************************************
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # we have to reshape our images from 100, 1, 28, 28 to 100, 784 i.e. batch_size, input_size or feature_size
        images = images.reshape(-1, 28*28)
        #images = images.reshape(-1, 28*28).to(device)  # if gpu is available
        #labels = labels.to(device) # if gpu is available

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss={loss.item()}")



# iv. TESTING AND EVALUATION
#************************************************************************************************************
with torch.no_grad(): # we dont want to compute the gradient for all the steps we do
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)

        # predictions using optimized model parameters
        outputs = model(images)

        # torch max returns the value and index and we are interested in index since it represents the class labels
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct/n_samples
    print(f"accuracy = {acc}")


