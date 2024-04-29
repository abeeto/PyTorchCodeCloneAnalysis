import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
import Models
import ResNet18

#print("PyTorch Version: ", torch.__version__)
#print("Torchvision Version: ", torchvision.__version__)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__=='__main__':
    num_epoch = 10
    if_train = True
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    print(device)
    #net = Models.ForwardNet()
    net = ResNet18.ResNet18()
    net.to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if if_train:
        # Training Loop
        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(trainLoader, 0):
                # get the inputs
                inputs, labels = data
                #inputs = inputs.type(torch.FloatTensor)
                #labels = labels.type(torch.FloatTensor)
                # Wrap them in Variables
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

                # set gradient as 0
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() # instead of loss[0].data
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print("Finished Training")
    """
    dataiter = iter(testLoader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    """

    correct = 0
    total = 0
    labels = labels.to(device)
    for data in testLoader:
        images, labels = data
        images = images.to(device)  # missing line from original code
        labels = labels.to(device)  # missing line from original code
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


    """
    x = torch.randn(3)
    x = Variable(x, requires_grad=True)
    #print(x)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    #print(y)
    gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
    y.backward(gradients)

    #print(x.grad)

    # Set input Tensor
    input = torch.randn(1, 1, 32, 32) # 1 row * 1 image * 32 width * 32 width
    # Net init and input
    net = Models.ForwardNet()
    output = net(input)
    print(output.data)

    
    # Reset Gradients as zero
    net.zero_grad()
    output.backward(torch.randn(1, 10))
    print(input.grad)
    
    # Set a dummy target
    out = net(input)
    target = Variable(torch.arange(1, 11))
    criterion = nn.MSELoss()

    loss = criterion(out, target.float())

    print(loss.data)

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU 

    loss.backward()
    print(net.conv1.bias.grad)
    print(net.conv2.bias.grad)
    """
