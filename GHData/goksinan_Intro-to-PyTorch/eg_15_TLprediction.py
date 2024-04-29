import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import helper
from collections import OrderedDict
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg', warn=False, force=True)

# Get data
data_dir = 'Cat_dog_data'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Sanity check
images, labels = next(iter(trainloader))
#helper.imshow(images[0], normalize=True)

images, labels = next(iter(testloader))
#helper.imshow(images[0], normalize=True)

# Let's use one of the pre-trained models from the PyTorch library
# We will use DenseNet121. There are 121 layers
model = models.densenet121(pretrained=True) # pretrained network will be downloaded
# Let's take a look
model
# This network is trained for ImageNet dataset. With its current classifier it won't work with our classification
# problem.
model.classifier
# We will keep the features as they are, but we will replace the Fully Connected layer with our own
# and optimize this new FC's parameters during training. So we need to freeze all the layers before the classifier
# Freeze our feature parameters:
for param in model.parameters():
    param.requires_grad = False # The gradients will not be calculated

# Let's build our own classifier, a fully connected network with 2 hidden layers
# Use Sequential with OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024,500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500,2)),
    ('output', nn.LogSoftmax(dim=1))]))

model.classifier  = classifier

# Out modified DenseNet is ready to use. However, since it is too deep, it will take a long time to train this
# netowrk on a regular cpu. Thus, we will use the gpu to speed up the process. Gpu can compute the same taks
# 100 times faster than the cpu. To demonstrate that, let's train the network for a few epochs and compare the
# execution times

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) # optimize only the fullcy connected later

cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
else:
    model.cpu()

epochs = 30
train_losses, test_losses = [], []

for e in range(epochs):
    # Training pass
    running_loss = 0
    start = time.time()
    for inputs, labels in trainloader:
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    else:
        # Report batch execution time
        print("CUDA = {}; Time per batch: {:.3f} seconds".format(cuda, (time.time() - start) / len(trainloader)))
        test_loss, accuracy = 0, 0
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = Variable(inputs), Variable(labels) # Looks like an unnecessary step
                if cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                log_ps = model(inputs) # generate scores
                test_loss += criterion(log_ps, labels) # compute loss
                ps = torch.exp(log_ps) # probabilities
                top_p, top_class = ps.topk(1, dim=1) # pick the class with the highest probability
                equals = top_class == labels.view(*top_class.shape) # compare to ground truth
                accuracy += torch.mean(equals.type(torch.FloatTensor)) # compute accuracy

        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}.. ".format(accuracy / len(testloader)))
        print("CUDA = {}; Time per epoch: {:.3f} seconds".format(cuda, time.time() - start))

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.legend(frameon=False)