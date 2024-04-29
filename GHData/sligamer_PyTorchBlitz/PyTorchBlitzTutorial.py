# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# Reference
# This tech report (Chapter 3) describes the dataset and the methodogy 
# followed when collecting it in much greater detail. Please cite it if you intend to use this dataset.
# Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
from __future__ import print_function
import datetime
from datetime import datetime as dt
from datetime import timedelta as td
import sys
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

global datetimeFormat
datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'

''' What is PyTorch? '''
class WiPyTorch():

    # compare this code to timer on tutorial website
    startTime = dt.now().strftime(datetimeFormat)

    # Note #
    # An uninitialized matrix is declared, but does not contain definite known values
    # before it is used. When an uninitialized matrix is created, 
    # whatever values were in the allocated memory at the time will appear as the initial values.
    # Construct a 5x3 matrix, uninitialized:
    x = torch.empty(5, 3)
    print(x)

    # Construct a randomly initialized matrix:
    x = torch.rand(5, 3)
    print(x)

    # Construct a matrix filled zeros and of dtype long:
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    # Construct a tensor directly from data:
    x = torch.tensor([5.5, 3])
    print(x)

    # or create a tensor based on an existing tensor. 
    # These methods will reuse properties of the input tensor, 
    # e.g. dtype, unless new values are provided by user
    # new_* methods take in sizes
    x = x.new_ones(5, 3, dtype=torch.double)      
    print(x)

    # override dtype!
    # result has the same size
    x = torch.randn_like(x, dtype=torch.float)   
    print(x)                                      

    # Get its size:
    print(x.size())

    # Note #
    # torch.Size is in fact a tuple, so it supports all tuple operations.

    # Operations #
    # There are multiple syntaxes for operations. 
    # In the following example, we will take a look at the addition operation.

    # Addition: syntax 1
    y = torch.rand(5, 3)
    print(x + y)

    
    # Addition: syntax 2
    print(torch.add(x, y))

    #Addition: providing an output tensor as argument
    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print(result)

    # Addition: in-place
    # adds x to y
    y.add_(x)
    print(y)

    # Note #
    # Any operation that mutates a tensor in-place is post-fixed with an _. 
    # For example: x.copy_(y), x.t_(), will change x.

    # You can use standard NumPy-like indexing with all bells and whistles!
    print(x[:, 1])

    # Resizing: If you want to resize/reshape tensor, you can use torch.view:
    x = torch.randn(4, 4)
    y = x.view(16)

    # the size -1 is inferred from other dimensions
    z = x.view(-1, 8)  
    print(x.size(), y.size(), z.size())

    # If you have a one element tensor, use .item() to get the value as a Python number
    x = torch.randn(1)
    print(x)
    print(x.item())

    # Read later: https://pytorch.org/docs/torch
    # 100+ Tensor operations, including transposing, indexing, slicing, mathematical operations, 
    # linear algebra, random numbers, etc..

    # NumPy Bridge #    
    # Converting a Torch Tensor to a NumPy Array
    a = torch.ones(5)
    print(a)
    b = a.numpy()
    print(b)

    # See how the numpy array changed in value.
    a.add_(1)
    print(a)
    print(b)

    # Converting NumPy Array to Torch Tensor #
    # See how changing the np array changed the Torch Tensor automatically
    # import numpy as np
    # imported included as alias np 
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)

    # All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

    # CUDA Tensors #
    # let us run this cell only if CUDA is available
    # We will use ``torch.device`` objects to move tensors in and out of GPU
    if torch.cuda.is_available():
        # a CUDA device object
        device = torch.device("cuda")        

        # directly create a tensor on GPU
        y = torch.ones_like(x, device=device)  

        # or just use strings ``.to("cuda")``
        x = x.to(device)                       
        z = x + y
        print(z)

        # ``.to`` can also change dtype together!
        print(z.to("cpu", torch.double))    
        
    # Total running time of the script: ( 0 minutes 5.869 seconds)))
    endTime = dt.now().strftime(datetimeFormat)
    elaspedTime = dt.strptime(startTime, datetimeFormat)\
    - dt.strptime(endTime, datetimeFormat)
    print('Total running time of the script: ( {0} minutes {1} seconds)'.format(elaspedTime.min, elaspedTime.seconds))

'''Autograd: Automatic Differentiation'''
class AAD():
    
    # compare this code to timer on tutorial website
    startTime = dt.now().strftime(datetimeFormat)

    # Create a tensor and set requires_grad=True to track computation with it
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    # Do a tensor operation
    y = x + 2
    print(y)  
    
    # y was created as a result of an operation, so it has a grad_fn.
    print(y.grad_fn)

    # Do more operations on y
    z = y * y * 3
    out = z.mean()
    
    print(z, out)

    # .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place. 
    # The input flag defaults to False if not given.
    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)


    # Gradients
    # Let’s backprop now. Because out contains a single scalar, 
    # out.backward() is equivalent to out.backward(torch.tensor(1.)).
    out.backward()

    # Print gradients d(out)/dx
    print(x.grad)

    # Generally speaking, torch.autograd is an engine for computing vector-Jacobian product.
    # This characteristic of vector-Jacobian product makes it very convenient to 
    # feed external gradients into a model that has non-scalar output.
    # Now let’s take a look at an example of vector-Jacobian product:
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    print(y)

    # Now in this case y is no longer a scalar. torch.autograd could not compute the full Jacobian directly, 
    # but if we just want the vector-Jacobian product, simply pass the vector to backward as argument:
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)

    print(x.grad)

    # You can also stop autograd from tracking history on Tensors with 
    # .requires_grad=True either by wrapping the code block in with torch.no_grad():

    print(x.requires_grad)
    print((x ** 2).requires_grad)

    with torch.no_grad():
        print((x ** 2).requires_grad)

    # Or by using .detach() to get a new Tensor with 
    # the same content but that does not require gradients:
    print(x.requires_grad)
    y = x.detach()
    print(y.requires_grad)
    print(x.eq(y).all())

    # Read Later:
    # Document about autograd.Function is at https://pytorch.org/docs/stable/autograd.html#function
    
    # Total running time of the script: ( 0 minutes 3.757 seconds)
    endTime = dt.now().strftime(datetimeFormat)
    elaspedTime = dt.strptime(startTime, datetimeFormat)\
    - dt.strptime(endTime, datetimeFormat)
    print('Total running time of the script: ( {0} minutes {1} seconds)'.format(elaspedTime.min, elaspedTime.seconds))

''' Neural Networks - Single Channel'''
class NN_SingleChannel(nn.Module):

    # compare this code to timer on tutorial website
    startTime = dt.now()

    # Neural networks can be constructed using the torch.nn package.
    # Now that you had a glimpse of autograd, nn depends on autograd to define models and differentiate them. 
    # An nn.Module contains layers, and a method forward(input)that returns the output.
    # It is a simple feed-forward network. It takes the input, 
    # feeds it through several layers one after the other, and then finally gives the output.

    # A typical training procedure for a neural network is as follows:

    # Define the neural network that has some learnable parameters (or weights)
    # Iterate over a dataset of inputs
    # Process input through the network
    # Compute the loss (how far is the output from being correct)
    # Propagate gradients back into the network’s parameters
    # Update the weights of the network, 
    # typically using a simple update rule: weight = weight - learning_rate * gradient

    # Define the network
    # Let’s define this network:
    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    def __init__(self):
        super(NN_SingleChannel, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        # 6*6 from image dimension
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

''' Neural Networks - Multi-Channel'''
class NN_MultiChannel(nn.Module):

    # Define the network
    # Let’s define this network:
    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    def __init__(self):
        super(NN_MultiChannel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

''' Class to run/output NN ''' 
class RunNN():

    # compare this code to timer on tutorial website
    startTime = dt.now().strftime(datetimeFormat)
    global net
    net = NN_SingleChannel()
    print(net)

    # You just have to define the forward function, and the backward function 
    # (where gradients are computed) is automatically defined for you using autograd. 
    # You can use any of the Tensor operations in the forward function.

    # The learnable parameters of a model are returned by net.parameters()
    params = list(net.parameters())
    print(len(params))
    # conv1's .weight
    print(params[0].size())  

    # Let’s try a random 32x32 input. Note: expected input size of this net (LeNet) is 32x32. 
    # To use this net on the MNIST dataset, please resize the images from the dataset to 32x32.
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    # Zero the gradient buffers of all parameters and backprops with random gradients:
    net.zero_grad()
    out.backward(torch.randn(1, 10))
    

    # Note #
    # torch.nn only supports mini-batches. 
    # The entire torch.nn package only supports inputs that are a mini-batch of samples, 
    # and not a single sample.

    # For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
    #If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
    # Before proceeding further, let’s recap all the classes you’ve seen so far.

    # Recap:

    # torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). 
    # Also holds the gradient w.r.t. the tensor.
    # nn.Module - Neural network module. Convenient way of encapsulating parameters, 
    # with helpers for moving them to GPU, exporting, loading, etc.

    # nn.Parameter - A kind of Tensor, 
    # that is automatically registered as a parameter when assigned as an attribute to a Module.

    # autograd.Function - Implements forward and backward definitions of an autograd operation. 
    # Every Tensor operation creates at least a single Function node that connects 
    # to functions that created a Tensor and encodes its history.

    # At this point, we covered: Defining a neural network
    # Processing inputs and calling backward

    # Still Left: Computing the loss
    # Updating the weights of the network

    # Loss Function #

    # A loss function takes the (output, target) pair of inputs, 
    # and computes a value that estimates how far away the output is from the target.
    # There are several different loss functions under the nn package. 
    # A simple loss is: nn.MSELoss which computes the mean-squared error between the input and the target.
    output = net(input)
    # a dummy target, for example
    target = torch.randn(10)  

    # make it the same shape as output
    target = target.view(1, -1)  
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    # Now, if you follow loss in the backward direction, using its .grad_fn attribute, 
    # you will see a graph of computations that looks like this:

    #input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> view -> linear -> relu -> linear -> relu -> linear -> MSELoss -> loss

    # So, when we call loss.backward(), the whole graph is differentiated w.r.t. the loss, 
    # and all Tensors in the graph that has requires_grad=True 
    # will have their .grad Tensor accumulated with the gradient.

    # For illustration, let us follow a few steps backward:
    # MSELoss
    print(loss.grad_fn)  
    # Linear
    print(loss.grad_fn.next_functions[0][0])  
    # ReLU
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  

    # Backprop #

    # To backpropagate the error all we have to do is to loss.backward(). 
    # You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
    # Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.

    # zeroes the gradient buffers of all parameters
    net.zero_grad()     

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # Now, we have seen how to use loss functions.
    # Read Later: #

    # The neural network package contains various modules and loss functions 
    # that form the building blocks of deep neural networks. A full list with documentation is here.

    # The only thing left to learn is: Updating the weights of the network

    # Update the weights #

    # The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
    # weight = weight - learning_rate * gradient
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    # However, as you use neural networks, you want to use various different update rules such as 
    # SGD, Nesterov-SGD, Adam, RMSProp, etc. 
    # To enable this, we built a small package: torch.optim that implements all these methods. 
    # Using it is very simple:

    # import torch.optim as optim

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    
    # zero the gradient buffers
    optimizer.zero_grad()   
    output = net(input)
    loss = criterion(output, target)
    loss.backward()

    # Does the update
    optimizer.step()    
    
    # Note #
    # Observe how gradient buffers had to be manually set to zero using optimizer.zero_grad(). 
    # This is because gradients are accumulated as explained in the Backprop section.

    # Total running time of the script: ( 0 minutes 3.766 seconds)
    endTime = dt.now().strftime(datetimeFormat)
    elaspedTime = dt.strptime(startTime, datetimeFormat)\
    - dt.strptime(endTime, datetimeFormat)
    print('Total running time of the script: ( {0} minutes {1} seconds)'.format(elaspedTime.min, elaspedTime.seconds))

''' Training a Classifier '''
class TAC():
    
    # compare this code to timer on tutorial website
    startTime = dt.now().strftime(datetimeFormat)

    # This is it. You have seen how to define neural networks, 
    # compute loss and make updates to the weights of the network.

    # What about data? #

    # Training an image classifier #

    # We will do the following steps in order:

    # 1. Load and normalizing the CIFAR10 training and test datasets using torchvision
    # 2. Define a Convolutional Neural Network
    # 3. Define a loss function
    # 4. Train the network on the training data
    # 5. Test the network on the test data

    # 1. Loading and normalizing CIFAR10
    #import torch
    # CPU-Only
    # pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
    # Cuda
    # pip install torch==1.3.1 torchvision==0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
    #import torchvision
    #import torchvision.transforms as transforms
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # research num_workers on non Cuda systems?
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    #global
    global classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Let us show some of the training images, for fun.

    #import matplotlib.pyplot as plt
    #import numpy as np

    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    global images, labels
    images, labels = dataiter.next()

    # show images    
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 2. Define a Convolutional Neural Network
    # Copy the neural network from the Neural Networks section before and modify it to take 3-channel images 
    # (instead of 1-channel images as it was defined).
    # See class NN_MultiChannel
    net = NN_MultiChannel()
    print(net)
    # 3. Define a loss function
    # Let’s use a Classification Cross-Entropy loss and SGD with momentum.
    #import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the network on the training data
    # This is when things start to get interesting. 
    # We simply have to loop over our data iterator, and feed the inputs to the network and optimize.

    # loop over the dataset multiple times
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print every 2000 mini-batches
            if i % 2000 == 1999:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Let’s quickly save our trained model:
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # See here for more details on saving PyTorch models.
    # https://pytorch.org/docs/stable/notes/serialization.html

    # 5. Test the network on the test data
    # We have trained the network for 2 passes over the training dataset. 
    # But we need to check if the network has learnt anything at all.
    # We will check this by predicting the class label that the neural network outputs, 
    # and checking it against the ground-truth. If the prediction is correct, 
    # we add the sample to the list of correct predictions.
    # Okay, first step. Let us display an image from the test set to get familiar.
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net.load_state_dict(torch.load(PATH))

    #Okay, now let us see what the neural network thinks these examples above are:
    outputs = net(images)
    global _, predicted
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

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
    
    global class_correct, class_total
    
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

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)

    net.to(device)

    inputs, labels = data[0].to(device), data[1].to(device)

    # Note #    
    #Why dont I notice MASSIVE speedup compared to CPU? Because your network is really small.
    #  Exercise: #
    #  Try increasing the width of your network (argument 2 of the first nn.Conv2d, and argument 1 of the second nn.Conv2d – they need to be the same number), 
    #  see what kind of speedup you get.

    # Goals achieved: #

    ## Understanding PyTorch’s Tensor library and neural networks at a high level.  ##
    # Train a Training a Classifier
    # Total running time of the script: ( 0 minutes 3.766 seconds)
    endTime = dt.now().strftime(datetimeFormat)
    elaspedTime = dt.strptime(startTime, datetimeFormat)\
    - dt.strptime(endTime, datetimeFormat)
    print('Total running time of the script: ( {0} minutes {1} seconds)'.format(elaspedTime.min, elaspedTime.seconds))

''' Main Startup '''
def main():
    
    # how long does this take to run
    startTime = dt.now().strftime(datetimeFormat)

    # Run What is PyTorch?
    WiPyTorch()
    # Run Autograd: Automatic Differentiation?
    AAD()
    # Run the Neural Network
    NN_SingleChannel()
    NN_MultiChannel()

    # Run the Training a Classifier
    TAC()

    # Total running time of the script: ( 0 minutes 3.766 seconds)
    endTime = dt.now().strftime(datetimeFormat)
    elaspedTime = dt.strptime(startTime, datetimeFormat)\
    - dt.strptime(endTime, datetimeFormat)
    print('Total running time of the tutorial blitz script: ( {0} minutes {1} seconds)'.format(elaspedTime.min, elaspedTime.seconds))

    # resource cleanup
    sys.exit(0)

if __name__ == "__main__":
    main()
