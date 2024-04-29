import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import train


def to_var(x):
    return Variable(x)

def evaluate():
    # loading the model
    net = torch.load("model.pkl")
    #loading the test set
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = dsets.MNIST(root='./data',train=False,transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=hw1.batch_size,shuffle=False)

    #test_dataset=train.test_dataset
    #test_loader=train.test_loader
    total_test = 0
    correct_test = 0
    avg_test=[]
    epochs_plt = []
    # testing
    for i, (images, labels) in enumerate(test_loader):
        images = to_var(images.view(-1, 28 * 28))
        labels = to_var(labels)
        out = net.forward(images)
        _, predicted = torch.max(out.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum()
    avg_test.append(correct_test/total_test)
    return np.mean(avg_test)
