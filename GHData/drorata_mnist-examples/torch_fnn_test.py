"""
Testing the network trained in torch_fnn_train.py

Code for this training is taken from:
https://www.kdnuggets.com/2018/02/simple-starter-guide-build-neural-network.html
"""

import numpy as np
from sklearn.metrics import classification_report
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch_fnn as fnn

print('Setting hyper parameters')
input_size = 784       # The image size = 28 x 28 = 784
hidden_size = 500      # The number of nodes at the hidden layer
num_classes = 10       # The number of output classes. In this case: 0 to 9
num_epochs = 5         # The number of times entire dataset is trained
batch_size = 100       # The size of input data took for one iteration
learning_rate = 0.001  # The speed of convergence
batch_size = 100       # The size of input data took for one iteration

print('Loading persisted network')
net = fnn.Net(input_size, hidden_size, num_classes)
net.load_state_dict(torch.load('./models/torch_fnn.dict'))

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

print('Starting predictions...')
correct = 0
total = 0
labels_arr = []
predicted_arr = []
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = net(images)
    # Choose the best class from the output: The class with the best score
    _, predicted = torch.max(outputs.data, 1)
    # Increment the total count
    total += labels.size(0)
    # Increment the correct count
    correct += (predicted == labels).sum()
    labels_arr += labels.numpy().tolist()
    predicted_arr += predicted.numpy().tolist()
print('Starting predictions... DONE')

labels_arr = np.array(labels_arr)
predicted_arr = np.array(predicted_arr)

print('Accuracy of the network on the 10K test images: {} %'.format(
    100 * correct / total))
print(classification_report(labels_arr, predicted_arr))
