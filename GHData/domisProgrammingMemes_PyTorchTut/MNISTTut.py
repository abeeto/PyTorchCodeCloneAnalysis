# unsing the Neural Network from NeuralNetworksTut für MNIST Classification
from __future__ import print_function

if __name__ == "__main__":
    import torch

    # Data Loading and normalizing using torchvision
    import torchvision
    import torchvision.transforms as transforms
    # use torch.nn for neural networks and torch.nn.functional for functions!
    import torch.nn as nn
    import torch.nn.functional as F
    # import torch optim for optimizer
    import torch.optim as optim

    # Path to save and load model
    net_path = './models/MNIST_net.pth'
    # Path for Data
    data_path = './data'

    # set up the divice (GPU or CPU) via input prompt
    cuda_true = input("Use GPU? (y) or (n)?")
    if cuda_true == "y":
        device = "cuda"
    else:
        device = "cpu"
    print("Device:", device)

    # Hyperparameters
    num_epochs = 10
    train_batch_size = 64
    test_batch_size = 64
    learning_rate = 0.001
    momentum = 0.5


    ## If running on Windows and you get a BrokenPipeError, try setting
    # the num_worker of torch.utils.data.DataLoader() to 0.
    # transform = transforms.transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.1307, ), (0.3081, ))],
    # )

    # Normalization on the pictures
    normalize = transforms.Normalize(mean=0.5, std=1)               # this does nothing as the tensors already are normalized this way

    transform = transforms.transforms.Compose(
        [transforms.ToTensor(),
         normalize]
    )

    #transform = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(root=data_path, train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root=data_path, train=True,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=True, num_workers=0)


    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
                   '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    import matplotlib.pyplot as plt

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(npimg[0], cmap="gray")
        plt.title("Example")
        plt.show()

    def showexample(imgages):
        # show images for fun!!
        # print lables before show picture otherwise the programm will not contunue
        print("Example labels:", "".join("%5s," % classes[labels[j]] for j in range(10)))
        # make a grid with utils!
        imshow(torchvision.utils.make_grid(images))

    # get some random training images in a batch with batch_size!
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    showexample(images)
    print()

    # lets define a network: (always as class!)
    class Net(nn.Module):
        # always need the init with super!
        def __init__(self):
            super(Net, self).__init__()
            # kernel
            # 1 input image channel, 6 output channels, 3x3 square convolution (3 is the filter which typically is 3 or 5)
            self.conv1 = nn.Conv2d(1, 6, 3)
            # first of conv2 has to be last of conv1!
            self.conv2 = nn.Conv2d(6, 16, 3)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120)   # image dimension is: 5x5
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # max pooling over a (2, 2) window
            # print(x.size(), "this is the size!!!!!")
            # x = self.conv1(x)
            # print(x.size(), "this is the size!!!!!")
            # x = F.relu(x)
            # print(x.size(), "this is the size!!!!!")
            # x = F.max_pool2d(x, 2)
            # print(x.size(), "this is the size!!!!!")
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # print(x.size())
            # if the size is a square you can only specify a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            # get size for linear layer
            # print(" ", x.size())
            # exit()
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    load = input("Load Network? (y) or (n)?")
    if load == "y":
        net.load_state_dict(torch.load(net_path))
    else:
        pass

    net.to(device=device)
    print(net)


    # define a loss function and optimizer
    # TODO: when to use which criterion? which optimizer (SGD, ...); what is momentum?
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


    # train the network
    # This is when things start to get interesting. We simply have to loop over our data iterator
    # and feed the inputs to the network and optimize.
    def train_network(epochs: int):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):                           # i is equal to the batch_number (index/batch_idx)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print stats
                running_loss += loss.item()
                if i % 100 == 100 - 1:             # print every 100 batches
                    print("[%d, %d] loss: %.5f" %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print("Training Finished!")


    # execute training!
    train_true = input("Train network? (y) or (n)?")
    if train_true == "y":
        train_network(num_epochs)

        # save the network?
        save = input("Save net? (y) or (n)?")
        if save == "y":
            torch.save(net.state_dict(), net_path)
        else:
            pass

    else:
        pass

    # test the network on test data
    def test(images, labels, testnum: int = 0):
        if testnum == 1:
            # print difference groundtruth to predicted
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    # images, labels = data[0], data[1]
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images: %5f %%' % (
                100 * correct / total))

        elif testnum == 2:
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(10):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))

        else:
            # print images
            print("Groundtruth: ", " ".join("%5s," % classes[labels[j]] for j in range(64)))
            # imshow(torchvision.utils.make_grid(images))

            net.load_state_dict(torch.load(net_path))
            outputs = net(images)

            _, predicted = torch.max(outputs, 1)
            print("Predicted: ", " ".join("%5s," % classes[predicted[j]] for j in range(64)))
            # imshow(torchvision.utils.make_grid(images))


    # test(images, labels, 1)
    # test(images, labels, 2)

    from PIL import Image

    # just evaluate the net
    net.eval()

    # ------------------------------------------------------------------------------
    # load in own data
    def val_own_data():

        # does only work properly on 28/28 images to begin with
        y = transforms.transforms.Compose(
            [transforms.Resize((28, 28)),
             transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)
             ]
        )
        valdata = {}
        testim = Image.open(r"C:\Users\domi\Desktop\handdigits\hand3.png")
        t_testim = y(testim)
        # abc = torchvision.transforms.ToPILImage()(t_testim)
        plt.imshow(t_testim.permute(1, 2, 0), cmap="gray")
        plt.show()
        t_testim = t_testim.unsqueeze(0).to(device)
        valdata[3] = t_testim

        testim = Image.open(r"C:\Users\domi\Desktop\handdigits\hand5.png")
        t_testim = y(testim)
        # abc = torchvision.transforms.ToPILImage()(t_testim)
        plt.imshow(t_testim.permute(1, 2, 0), cmap="gray")
        plt.show()
        t_testim = t_testim.unsqueeze(0).to(device)
        valdata[5] = t_testim

        with torch.no_grad():
            for label in valdata:
                valdata[label]
                data = net(valdata[label]).to(device)
                print(data.data.max(1, keepdim=True)[1])
        exit()
        # ------------------------------------------------------------------------------

    # WORKS!!
    examples = enumerate(testloader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    with torch.no_grad():
        net.to(device)
        output = net(example_data).to(device)
    for j in range(20):
        plt.subplot(5, 4, j + 1)
        plt.tight_layout()
        plt.imshow(example_data[j][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}\nGround truth: {}".format(
            output.data.max(1, keepdim=True)[1][j].item(), example_targets[j]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # test(images, labels)

# spike for CUDA AFTER programm is done... why??
