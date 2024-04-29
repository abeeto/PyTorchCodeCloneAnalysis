import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import MyNet

### HYPERPARAMETERS
### TODO: Play around with these for different effects on training!

# Number of times to repeat the training loop.
num_epoch = 10

# How many images to feed into the network at a time. Lower is more accurate,
# but much slower to train. Should not exceed 50000! (the size of the image set)
batch_size = 10000

# How quickly you want the network to attempt to converge. Smaller values are
# more accurate but much slower, while higher values can actually make the
# network perform worse over time. Try to find a happy medium.
learning_rate = 0.001

# The name of the file that will store the neural net when training completes.
net_name = "my_neural_net"


### Load the CIFAR10 dataset. Consists of 50000 images that are classified into
### 10 different classes of various objects.

# The transform function that converts an image to a PyTorch tensor and
# normalizes it.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


### Initialize the neural net and begin the training loop.
net = MyNet()

# NOTE: Cross-entropy loss is the default for classification, but try looking
# into other loss functions that could be used.
criterion = nn.CrossEntropyLoss()

# NOTE: Check out the torch.optim library for other optimization algorithms that
# could be used instead.
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# Begin the training loop.
for epoch in range(num_epoch):
    # Helps keep track of where you are in training. Don't like this method?
    # Check out the tqdm module, which will print a loading bar for for loops.
    print("Epoch:", epoch+1)

    for data in trainloader:
        # Get the inputs with labels
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Feedforward
        #inputs = inputs.view(-1, 784)
        outputs = net(inputs)

        # Backpropogation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the trained neural network.
torch.save(net.state_dict(), "./" + net_name + ".pth")
print("Finished training! Saved model parameters in file " + net_name + ".pth")
