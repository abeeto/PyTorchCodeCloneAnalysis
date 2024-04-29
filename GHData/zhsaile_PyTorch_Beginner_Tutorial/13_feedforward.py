import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 0) Preparation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784  #28x28
hidden_size = 100
num_classes = 10
epochs = 2
batch_size = 100
lr = 0.001

# 1) Data import
transform = transforms.Compose([transforms.ToTensor()])
train_datasets = torchvision.datasets.MNIST(root='./data', train=True,
        transform=transform, download=True)

test_datasets = torchvision.datasets.MNIST(root='./data', train=False,
        transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
        batch_size=batch_size, shuffle=False)
examples = iter(train_loader)
images, labels = examples.next()
print(images.shape)

# 2) Construct model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

model = Model(input_size, hidden_size, num_classes)

# 3) Construct loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 4) Training loop
n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # reshape the raw image data which is 100x1x28x28 to 100x784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1} / {epochs}, step {i+1}/{n_total_steps}, loss={loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        print(outputs)
        # max return value and index
        _, predictions = torch.max(outputs, 1)
        print('pred: ',predictions)
        print('labels: ', labels)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        break

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')


examples = iter(train_loader)
samples, labels = examples.next()
#print(samples.shape, labels)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()
