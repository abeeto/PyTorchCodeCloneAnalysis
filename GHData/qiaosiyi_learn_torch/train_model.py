import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from mymodel import *

from torch.utils.tensorboard import SummaryWriter
myWriter = SummaryWriter('../logs_train_alexnet_high/')

myTransforms = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.Resize((128,128)),
    # transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=myTransforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=myTransforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)


myModel = torchvision.models.alexnet(pretrained=True)
myModel.classifier = nn.Sequential(
    nn.Linear(512*7*7, 512),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(512, 128),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(128, 10)
)
# myModel = Tudui128()
# myModel = Tudui()
# inchannel = myModel.fc.in_features
# myModel.fc = nn.Linear(inchannel, 10)


myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)

learning_rate = 0.001
myOptimizier = optim.SGD(myModel.parameters(), lr = learning_rate, momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

for _epoch in range(40):
    training_loss = 0.0
    for _step, input_data in enumerate(train_loader):
        image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)
        predict_label = myModel.forward(image)

        loss = myLoss(predict_label, label)

        myWriter.add_scalar('training loss', loss, global_step= _epoch*len(train_loader) + _step)

        myOptimizier.zero_grad()
        loss.backward()
        myOptimizier.step()

        training_loss = training_loss + loss.item()
        if _step % 100 == 0:
            print('[iteration - %3d] training loss: %.3f' % (_epoch*len(train_loader) + _step, training_loss/10))
            training_loss = 0.0
            print()
    correct = 0
    total = 0

    myModel.eval()
    for images, labels in test_loader:

        images = images.to(myDevice)
        labels = labels.to(myDevice)
        outputs = myModel(images)
        numbers, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    print('Testing Accuracy: %.3f %%' % ( 100 * correct / total))
    myWriter.add_scalar('test_accuracy', 100 * correct / total, _epoch + 1)



























