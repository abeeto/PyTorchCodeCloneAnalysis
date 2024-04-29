import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

train_dataset = dsets.MNIST(root='./data/MNIST',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False)

test_dataset = dsets.MNIST(root='./data/MNIST',
                           train=False,
                           transform=transforms.ToTensor())

batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=5, stride=1, padding=0)

        # Max pool 2
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        # Fully connected (readout)
        self.fc1 = nn.Linear(32 * 4 * 4, 32 * 4 * 4)
        self.fc2 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu(out)

        # Max pool 1
        out = self.maxpool(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu(out)

        # Max pool 2
        out = self.maxpool(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

model=CNNModel()

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

    # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

    # Forward pass to get output/logits
        outputs = model(images)

    # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
        loss.backward()

    # Updating parameters
        optimizer.step()

        iter += 1
        if iter %100 == 0:
            print('Iteration: {}. Loss: {} '.format(iter, loss.data[0]))

        if iter % 500 == 0:
        # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images)
                    labels = Variable(labels)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                correct += (predicted.cpu() == labels.cpu()).sum()


                accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))
