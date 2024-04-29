import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 60, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input, 16 output, 3x3 square kernal
        self.conv2 = nn.Conv2d(60, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_featurres = 1
        for s in size:
            num_featurres *= s
        return num_featurres


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    # What is PyTorch?
    '''
    # Construct an empty 5x3 matrix
    x = torch.empty(5, 3)
    # Construct a randomly init 5x3 matrix
    x = torch.rand(5, 3)
    # Construct a 5x3 matrix of zeros
    x = torch.zeros(5, 3, dtype=torch.long)
    # Manually construct a 5x3 matrix
    x = torch.tensor([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    # Construct new tensor - takes in new sizes (x.size())
    x = x.new_ones(5, 3, dtype=torch.double)
    # Override type
    x = torch.randn_like(x, dtype=torch.float)
    # Element wise Addition
    y = torch.randn_like(x, dtype=torch.float)
    print(x + y)
    # ... or ...
    print(torch.add(x, y))
    # ... or ... place output into another tensor
    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print(result)
    # ... or ... inplace addition to y
    y.add_(x)
    # indexing like numpy
    print(x[:, 1])
    # establish the shape of a tensor
    z = x.view(-1, 5)  # takes the shape of x (5, 3) = 15 and since one of z is 5 then the other is 3
    print(z.size())
    # get the number within a tensor if there is only one
    print(torch.randn(1).item())
    # tensor to numpy array
    print(x.numpy())
    # numpy to torch tensor
    a = np.ones(5)
    b = torch.from_numpy(a)
    print(a)
    print(b)
    # CUDA Tensors
    x = torch.randn(5, 3)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))
    '''

    # Autograd: Automatic Gradient
    '''
    # tracks operations made on the tensor
    x = torch.ones(2, 2, requires_grad=True)
    # to stop tracking -> x.detach()

    y = x + 2
    z = y * y * 3
    out = z.mean()  # a single scalar
    print(z, out)
    out.backward()  # equ to out.backward(torch.tensor(1.))
    print(x.grad)  # d(out) / dx

    # if y_hat - f(x_hat) then the gradient of y_hat
    # ... with respect to x_hat is the Jacobian matrix
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    print(y)
    '''

    # Neural Networks
    '''
    net = Net()
    print(net)
    # getting learnable parameters
    params = list(net.parameters())
    print(params)

    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    # clear all gradient buffers and backprop with random gradients
    net.zero_grad()
    out.backward(torch.randn(1, 10))

    # loss functioins
    out = net(input)
    target = torch.randn(10)  # dummy target
    target = target.view(1, -1)  # make same shape as output
    criterion = nn.MSELoss()  # apply MSE loss function

    loss = criterion(out, target)  # MSE loss between output and target
    print(loss)

    # backprop loss through gradients
    net.zero_grad()  # must first clear gradient buffers
    loss.backward()  # propagate loss through

    # updating the weights
    opt = optim.SGD(net.parameters(), lr=0.01)
    opt.zero_grad()  # clear the gradient buffers
    out = net(input)
    loss = criterion(out, target)  # calcs the MSE loss
    loss.backward()  # propagates the loss
    opt.step()  # updates the weight
    '''

    # Training an image classifier
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck')

    # show some images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # define and build model
    PATH = './cifar_net.pth'
    # net = Net()
    # crit = nn.CrossEntropyLoss()
    # opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # # train the network
    # epochs = 5
    # for ep in range(epochs):
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #         # zero the parameter gradient buffers
    #         opt.zero_grad()
    #         # forward and backward and optimize
    #         out = net(inputs)
    #         loss = crit(out, labels)
    #         loss.backward()
    #         opt.step()
    #         # statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:  # printing every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' % (ep + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #
    # # save the trained model
    # torch.save(net.state_dict(), PATH)
    # # test the model on test data
    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    #
    # # print images
    # imshow(torchvision.utils.make_grid((images)))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # load model
    # net = Net()
    # net.load_state_dict(torch.load(PATH))
    # # get overall accuracy
    # correct, total = 0, 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    # # get accuracy by class
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]
    #     ))

    # Train on GPU (can also train on multiple GPU's
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # train the network
    epochs = 5
    for ep in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradient buffers
            opt.zero_grad()
            # forward and backward and optimize
            out = net(inputs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            # statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # printing every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (ep + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # save the trained model
    torch.save(net.state_dict(), PATH)



    # TODO (today): PyTorch practice all day, workout, PyTorch
    # TODO (Thurs): PyTorch all morning, interview, Project 8, workout, Project 8

