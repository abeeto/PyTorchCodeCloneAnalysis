"""
Train a Feedforward Neural Networks (FNN) on the MNIST dataset.
Code for this training is taken from:
https://www.kdnuggets.com/2018/02/simple-starter-guide-build-neural-network.html

After training the network is persisted. Testing the results is coded in
torch_fnn_test.py.
"""

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch_fnn as fnn


input_size = 784       # The image size = 28 x 28 = 784
hidden_size = 500      # The number of nodes at the hidden layer
num_classes = 10       # The number of output classes. In this case: 0 to 9
num_epochs = 5         # The number of times entire dataset is trained
batch_size = 100       # The size of input data took for one iteration
learning_rate = 0.001  # The speed of convergence

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


net = fnn.Net(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    # Load a batch of images with its (index, data, class)
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable: change image from a vector of size
        # 784 to a matrix of 28 x 28
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # Intialize the hidden weight to all zeros
        optimizer.zero_grad()
        # Forward pass: compute the output class given a image
        outputs = net(images)
        # Compute the loss: difference between the output class and the
        # pre-given label
        loss = criterion(outputs, labels)
        # Backward pass: compute the weight
        loss.backward()
        # Optimizer: update the weights of hidden nodes
        optimizer.step()

        if (i+1) % 100 == 0:                              # Logging
            print(
                'Epoch [{e}/{te}], Step [{s}/{ts}], Loss: {loss:.4f}'.format(
                    e=epoch+1,
                    te=num_epochs,
                    s=i+1,
                    ts=len(train_dataset)//batch_size,
                    loss=loss.data[0]
                ))

print('Persisting model...')
torch.save(net.state_dict(), './models/torch_fnn.dict')
print('Run "torch_fnn_test.py" to check the accuracy')
