import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x1 = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x2 = self.pool(torch.nn.functional.relu(self.conv2(x1)))
        x3 = x2.view(-1, 16 * 5 * 5)
        x4 = torch.nn.functional.relu(self.fc1(x3))
        x5 = torch.nn.functional.relu(self.fc2(x4))
        x6 = self.fc3(x5)

        return x6



if __name__ == "__main__":
    """
    1. Loading and normalizaing CIFAR10
    
    The output of torchvision datasets are PILImage images of range [0,1].
    We transform them to Tensors of normalized range [-1,1].
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """
    Let us show some of the training images, for fun
    """
    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    #
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    """
    2. Define a Convolutional Neural Network
    
    Copy the neural network from the Neural Networks section before and modify it to take 3-channel images
    (instead of 1-channel images as it was defined)
    """
    net = Net()


    """
    3. Define a Loss function and optimizer
    
    Let's use a Classification Cross-Entropy loss and SGD with momentum
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    """
    4. Train the network
    
    This is when things start to get interesting. We simply have to loop over our data iterator,
    and feed the inputs to the network and optimize.
    """
    net.train()
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = running_loss + loss.item()
            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f' % (epoch + 1,
                                                i + 1,
                                                running_loss / 2000))
                running_loss = 0.0

            break

    print('Finished Training')

    """
    Let's quickly save our trained model
    """
    # PATH = "./cifar_net.pth"
    # torch.save(net.state_dict(), PATH)


    """
    5. Test the network on the test data
    
    We will check this by predicting the class label that the neural network outputs,
    and checking it against the ground-truth.
    If the prediction is correct, we add the sample to the list of correct predictions
    """
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # net = Net()
    # net.load_state_dict(torch.load(PATH))

    net.eval()
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    net.eval()
    with torch.no_grad():
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] = class_correct[label] + c[i].item()
                class_total[label] = class_total[label] + 1

    for i in range(10):
        print('Accuracy of %5s : %3d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))