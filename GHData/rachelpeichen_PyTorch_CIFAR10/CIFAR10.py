#1. Loading and normalizing CIFAR10.
import torch
from torch.utils.data import Dataset, TensorDataset # Heart of PyTorch data loading utility
import torchvision # Use torchvision to load CIFAR10.
import torchvision.transforms as transforms # Use transforms to transform image.


# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

# Use transforms.Compose to chain transforms together
transform = transforms.Compose( 
    [transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#Let us show some of the training images just for fun
if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Show images
    imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 2. Define a Convolutional Neural Network
import torch.nn as nn # Use torch.nn to construct a neural network
import torch.nn.functional as F

# Take 3-channel images
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv layers: 3 input image channel, 6 output channels, kernal size of 5x5 square convolution 
        self.conv1 = nn.Conv2d(3, 6, 5) # Apply a 1D convlution over an input signal composed of several iput planes.
        self.conv2 = nn.Conv2d(6, 16, 5) # Apply a 2D convlution over an input signal composed of several iput planes.

        # Pooling layer: maximum pooling over a 2 x 2 window (kernel size = 2, stride = 2)
        self.pool = nn.MaxPool2d(2, 2) #Applies a 2D max pooling over an input signal composed of several input planes.
        
        # Linear layers: Applya linear transformation to the incoming data: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # ReLU layer is a non-linear activation function to give out the final value from a neuron, the result gives a range from 0 to infinity.
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        
        # Reshape the tensor: return a new tensor with same data as the self tensor but of a different shape.
        x = x.view(-1, 16 * 5 * 5) # -1: automatically calculate dimension 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()
print(net)

# 3. Define a Loss function and optimizer
import torch.optim as optim

# Use a Classification Cross-Entropy loss and SGD with momentum as our optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. Train the network
def train():
    torch.multiprocessing.freeze_support()
    for epoch in range(10):  # Loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH) # Save our trained model


if __name__ == "__main__":
    train()

# 5. Test the network on the test data
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # Print some images from the test data
    import numpy as np
    from matplotlib import pyplot as plt
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    # Load back 
    net = Net()
    PATH = './cifar_net.pth'
    net.load_state_dict(torch.load(PATH))
    
    # See what our neural network thinks these images above are:
    outputs = net(images)
    print(outputs)
    
    # Since the outputs are energies of 10 classes, lets get the index of the highest energy:
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(4)))

    # Look how our cnn performs on the whole dataset:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    # Check every classes' performance:
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)