import time
import torch
import matplotlib.pyplot as plt
from mnist_utils import mnist_train_loader
from mnist_utils import mnist_test_loader

import utils

#My Nets
from conv_net import ConvNet
from ff_net import FFNet


train_loader = mnist_train_loader(batch_size=5)
test_loader = mnist_test_loader(batch_size=5)

print('==>>> total training batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

num_epochs = 5

#Feed Forward Net
model = FFNet()

#Convolutional Net
#model = ConvNet()


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

start_time = time.time()

train_acc_history, test_acc_history = \
    utils.train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs)

training_time = time.time() - start_time
training_time = time.gmtime(training_time)

print("Total Training Time:",
      str(training_time.tm_hour) + ":" + str(training_time.tm_min) + ":" + str(training_time.tm_sec))

print("Training Set Accuracy:", train_acc_history[-1])
print("Test Set Accuracy:", test_acc_history[-1])

plt.plot(range(len(train_acc_history)), train_acc_history, label='Training Data')
plt.plot(range(len(test_acc_history)), test_acc_history, label='Test Data')

plt.title("Accuracies Throughout Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(loc='upper left')
plt.show()


