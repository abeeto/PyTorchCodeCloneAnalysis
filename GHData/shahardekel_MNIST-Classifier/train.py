import torch
import torch.nn.functional as tnnfunc
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import eval
import numpy as np

"""
short summary:
Inside the train file is the MNIST training process.
We got an input of 784 (28 pixels * 28 pixels), the training worked best for us with 1 hidden layer in size 64,
batch size of 100, learning rate = 1 and with 50 epochs.
We first set up a net with weights and biases (according to the size of the hidden layer).
We started the forward process where we used sigmoid on the hidden layer and softmax on the result.
Next, we performed the backward process, using the gradients we calculated in the theoretical part and the predicted
values ​​from the forward phase.
In the backward process we tried to learn with learning rate = 0.001, then with 0.01 and in the end we got
the best result with learning rate = 1.

To know the percentage of accuracy on the net, we performed a loop that goes through each epoch
(at first we tried sizes 10, 30 and in the end we got the best result with a size of 50).
For each epoch we calculated whether we were right in our prediction.
In the evaluate file we performed the same process, on the test set.

At the end we printed the average accuracy on the test set and two graphs describing the accuracy
of the sets in relation to the time (the epochs)
"""

# Hyper Parameters
input_size = 784 #28*28=784
hidden_size = 64
num_classes = 10
num_epochs = 50
batch_size = 100
learning_rate = 1

# train and test sets
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = dsets.MNIST(root='./data',train=False,transform=transform)

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# utils functions
def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def sigmoidPrime(s):
    # derivative of sigmoid
    # s: sigmoid output
    return s * (1 - s)


def softmax(X):
    exps = torch.exp(X)
    return (exps / torch.sum(exps))


class Neural_Network:
    def __init__(self, input_size, output_size, hidden_size):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.b1 = torch.zeros(self.hiddenSize)

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.b2 = torch.zeros(self.outputSize)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = sigmoid(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return softmax(self.z2)

    def backward(self, X, y, y_hat, lr=learning_rate):
        batch_size = y.size(0)
        y_hat1 = tnnfunc.one_hot(y_hat)
        y1 = tnnfunc.one_hot(y)

        if y1.shape[1] < 10:
            diff = 10 - y1.shape[1]
            add = torch.zeros(y1.shape[0], diff)
            y1 = torch.cat((y1, add), 1)

        if y_hat1.shape[1] < 10:
            diff = 10 - y_hat1.shape[1]
            add = torch.zeros(y_hat1.shape[0], diff)
            y_hat1 = torch.cat((y_hat1, add), 1)

        dl_dz2 = (1 / batch_size) * (y_hat1 - y1)
        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * sigmoidPrime(self.h)

        self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        _, predicted = torch.max(o.data, 1)
        self.backward(X, y, predicted)
        return predicted


def to_var(x):
    return Variable(x)


# building the net
net = Neural_Network(input_size, num_classes, hidden_size)



correct_train = 0
total_train = 0
correct_test = 0
total_test = 0
epochs_plt = []
train_plt = []
test_plt = []
avg_test=[]
# training
for epoch in range(num_epochs):
    epochs_plt.append(epoch)

    for i, (images, labels) in enumerate(train_loader):
        images = to_var(images.view(-1, 28 * 28))
        labels = to_var(labels)
        predicted = net.train(images, labels)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum()
    #train_plt.append(correct_train / total_train)


    for i, (images, labels) in enumerate(test_loader):
        images = to_var(images.view(-1, 28 * 28))
        labels = to_var(labels)
        out = net.forward(images)
        _, predicted = torch.max(out.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum()
    avg_test.append(correct_test / total_test)
    train_plt.append(correct_train / total_train)
    test_plt.append(correct_test / total_test)

torch.save(net, 'model.pkl')
"""avg = eval.evaluate_hw1()
print('avg=',avg)"""


#ploting the train and test graphs
plt.plot(epochs_plt,train_plt,color='blue', label='train set') #plt.plot(x,y)
plt.plot(epochs_plt,test_plt, color='red', label='test set')
plt.title('Accuracy As A Function Of Time (Epochs)')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

