#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from random import shuffle
import torch  
import torchvision  
import torchvision.transforms as transforms

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    batch_size = 4 

    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,batch_size,shuffle=True,num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=2)

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck') 

    import matplotlib.pyplot as plt 
    import numpy as np 

    #Function to Visualize the Data
    def imshow(img): 
        img = img /2+0.5 #This unnormalies the data
        npimg = img.numpy() 
        plt.imshow(np.transpose(npimg,(1,2,0))) 
        plt.show() 

    #Get some Random Training Images 
    detailer = iter(trainloader)
    images,labels = detailer.next() 

    #Show Images 
    imshow(torchvision.utils.make_grid(images))

    #Print Labels 
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    #Now we are going to define a Convolutional Neural Network. 
    import torch.nn as nn 
    import torch.nn.functional as F 

    class Net(nn.Module): 
        def __init__(self): 
            super().__init__() 
            self.conv1 = nn.Conv2d(3,6,5)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
            self.fc1 = nn.Linear(16*5*5,120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net() 

    #Define a Loss Function and Optimizer 
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 

    #Train the Network - loop over our data iterator, and feed the inputs to the network and optimize. 
    for epoch in range(2):  # loop over the dataset multiple times

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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training') 

    #Save the Trained Model 
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH) 

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    #Now load back in our saved model.
    #(note: saving and re-loading the model wasn’t necessary here, we only did it to illustrate how to do so)
    net = Net()
    net.load_state_dict(torch.load(PATH))

    #Now lets see what the NN thinks the above examples are. 
    outputs = net(images)

    #The outputs are energies for the 10 classes. The higher the energy for a class, the more the network thinks that the image is of the particular class. 
    # So, let’s get the index of the highest energy:
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'for j in range(4))) 

    #Now lets look at how the NN performs on the whole dataset.  
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %') 

    #At this point we are getting about 55% accurate classification. 
    #Which classes are the NN struggling with? 
    
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')