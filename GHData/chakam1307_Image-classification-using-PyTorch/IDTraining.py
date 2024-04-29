from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from model import SP_Classifier
from new_model import ID_Classifier

# CIFAR-10 Dataset
transformations = transforms.Compose([transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize((240, 320)),
                            transforms.Normalize(([0.5]), ([0.5]))
                            ])

batch_size = 10
number_of_labels = 10 
DEVICE = "cuda" if torch.cuda.is_available else "cpu"

train_set =CIFAR10(root="./data",train=True,transform=transformations,download=True)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader)*batch_size)

test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)


test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# MY DATASET
# batch_size = 10
# number_of_labels = 5
# DEVICE = "cuda" if torch.cuda.is_available else "cpu"


# trans = transforms.Compose([transforms.ToTensor(),
#                             transforms.Grayscale(num_output_channels=1),
#                             transforms.Resize((240, 320)),
#                             transforms.Normalize(([0.5]), ([0.5]))
#                             ])

# train_set = torchvision.datasets.ImageFolder(root="./dt/train", transform = trans)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
# print("The number of images in a training set is: ", len(train_loader)*batch_size)


# test_set = torchvision.datasets.ImageFolder(root="./dt/test", transform = trans)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
# print("The number of images in a test set is: ", len(test_loader)*batch_size)

# print("The number of batches per epoch is: ", len(train_loader))

# classes = train_set.classes






# model = SP_Classifier()
model = ID_Classifier()


loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

def testAccuracy():    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    accuracy = (100 * accuracy / total)
    return(accuracy)


def train(num_epochs):
    
    best_accuracy = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("The model will be running on", device, "device")
    
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            
            optimizer.step()

            
            running_loss += loss.item()     
            if i % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def testBatch():
    images, labels = next(iter(test_loader))

    imageshow(torchvision.utils.make_grid(images))
    
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))


# Function to test what classes performed well
def testClassess():
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # print(labels.shape)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(number_of_labels):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    
    train(20)
    print('Finished Training')

    # Test which classes performed well
    testClassess()
    
    # Let's load the model we just created and test the accuracy per label
    
    # model = SP_Classifier()
    model = ID_Classifier()
    path = "myFirstModel.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()