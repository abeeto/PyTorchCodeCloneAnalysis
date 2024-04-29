import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import NeuralNet
from torch import optim
# setting up datasets
trainset = datasets.MNIST('', download=True, train=True, transform = transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform = transforms.ToTensor())
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)
# network architecture
input_size = 784
hidden_size = [128, 64]
output_size = 10
# create the model
model = NeuralNet(input_size, hidden_size, output_size)
# define loss and optimisation
lossFunction = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
# training
num_epochs = 10
for epoch in range(num_epochs):
    loss_ = 0
    for images, labels in train_loader:
        images = images.reshape(-1, 784)

        output = model(images)

        loss = lossFunction(output, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_ += loss.item()

    print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(train_loader)))

#vand finally, tests
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 784)
        out = model(images)
        _, predicted = torch.max(out,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Testing accuracy: {} %".format(100 * correct/total))

#save the model
torch.save(model, 'mnist_model.pt')
