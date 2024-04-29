__author__ = 'KSM'

"""
Author: Kunjan Mhaske

This program trains the neural network consisting 2D CNN with linear functions.
It takes data from Fashion MNIST dataset and train the neural network with following:
Dataset Size: 60,000 Training and 10,000 Testing
Batch size = 100
Optimizer = Adam
Loss function = reLu
Activation Function = Softmax
Learning Rate = 0.001

The program gives output of overall accuracy of the network as well as respective class accuracies.
The program further plots the training loss curve against epochs as well as some random images
label predictions from the dataset and compare them with the ground truth values.
"""

import torch
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.ticker as ticker
import torchvision.datasets as datasets
import random

EPOCHS = 25
BATCH_SIZE = 100
LR = 0.001

# Defining the Neural Network
class Net(nn.Module):
    def __init__(self):
        '''
        This is initializer of the Net class
        '''
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)           # in channels, out channels, kernel size
        self.conv2 = nn.Conv2d(6,16,5)          # kernel size and stride

        self.fc1 = nn.Linear(16*5*5, 120)       # in features, out features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''
        This is forward logic of neural network
        :param x: dataset
        :return: Value with activation function applied to it
        '''
        x = F.avg_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        ''' view changes the dimensionality of the Tensor
            -1 represents not giving an explicit value and the
            function will figure out the correct value based
            on the input and the second parameter.
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x,dim=1)

if __name__ == '__main__':
    # apply transformation on each image data
    transform = transforms.Compose([transforms.Pad(2, padding_mode='constant'),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5],[0.5]) ])
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    mnist_trainset = datasets.FashionMNIST(root='./FashionMNISTdata', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    mnist_testset = datasets.FashionMNIST(root='./FashionMNISTdata', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    ####################################################################################################################
    ''' Using CUDA if available a GPU that supports it, else use the CPU
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device",device,torch.cuda.current_device())
    print("Properties:",torch.cuda.get_device_properties(device))

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    print("Optimizer:",optimizer)
    print("Training Start.....")
    print("Batch size:",BATCH_SIZE)
    # Training dataset
    saved_losses = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data                                   # Get the inputs
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()                                   # Zero the parameter gradient
                                                                    # useful for further iterations
            # forward + backward + optimizer
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            # print the losses for each epochs
            running_loss += loss.item()
        saved_losses.append(running_loss / 937)
        print('Epoch: ' + str(epoch + 1)+'/'+str(EPOCHS)+', Approximated Loss: ' + str(running_loss / 937))
        # 60,000 = total number of images in training  # 937 * 64 = 59968 viz. Approx: 60000

    print("___________________________Finished Training________________________________")

    ''' Saving the weights of the trained network to a text file that are human readable
    '''
    weights = list(net.parameters())
    with open('FmnistWeights_a1.txt','w') as f:
        for item in weights:
            f.write("%s \n"%item)

    model_out_path = "model_epoch_{}.pth".format(EPOCHS)
    torch.save(net, model_out_path)
    print("Model saved to {}".format(model_out_path))

    def plot(saved_losses):
        '''
        This method is used to plot the training loss vs epochs
        :param saved_losses: saved losses per epoch
        :return: None
        '''
        fig, ax = plt.subplots()
        x = np.linspace(1, EPOCHS, EPOCHS)
        saved_losses = np.array(saved_losses)
        ax.set_title("Average Model Training Loss over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Average Loss")

        tick_spacing = 5                # Adjust x-axis ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.plot(x, saved_losses, color='purple', marker=".")
        fig.savefig("Fmnist_training_loss_a1")
    plot(saved_losses)
    #######################################################################################
    print("________________________________________________________________________________")
    print("The sample comparison between ground truth vs the predicted labels")
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    print("\nGround Truth: ",' | '.join('%s'%classes[labels[j]] for j in range(10)))

    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print("Predicted:\t",' | '.join('%s'%classes[predicted[j]] for j in range(10)))
    #########################################################################################
    # Calculating the total accuracy of the network
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("\nAccuracy of network:",100*correct/total,"%")
    print()
    #######################################################################
    print("_______________________Accuracies per class______________________________")
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    class_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracies_y = []
    for i in range(10):
        accuracies_y.append(100*class_correct[i]/class_total[i])
        print("Accuracy of %s : %d %%"%(classes[i], 100*class_correct[i]/class_total[i]))

    print("---------------Plotting the predicted results of some random images-----------")
    def plot_sample_predictions(classes, X_test):
        '''
        This function plots the sample prediction labels of random images
        :param classes: list of classes
        :param X_test:  Input Data Loader
        :return: None
        '''
        class_ = 0
        images_per_row = 5
        rows = len(classes) // images_per_row

        dataiter = iter(X_test)
        img_name = 0
        for i in range(rows):
            fig, axis = plt.subplots(1, images_per_row)
            for i, axis in enumerate(axis):
                start = 0
                end = 10
                rand = random.randint(start, end)

                fig.set_size_inches(30,40)
                images, labels = dataiter.next()
                axis.text(0,45,'Correct:{}'.format(classes[labels[rand]]))
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                axis.text(0,40,'Predicted:{}'.format(classes[predicted[rand]]))

                img = np.asarray(images.cpu())
                # print(img.shape)
                nwimg = np.squeeze(np.asarray(img[:][:][rand]), axis=0)
                # print(nwimg.shape)
                axis.imshow(nwimg, cmap='gray')
                axis.axis('off')
                class_ += 1
            plt.savefig(str(img_name)+".png")
            img_name += 1
        plt.show()

    plot_sample_predictions(classes, test_loader)
    print("Figures saved in current working folder.")
    print("Done..")