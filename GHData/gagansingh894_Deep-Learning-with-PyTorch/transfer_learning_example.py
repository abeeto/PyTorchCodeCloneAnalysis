import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

data_dir = 'Cat_Dog_data'
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

model = models.densenet121(pretrained=True)

# freeze parameters
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

# GPU Code
for cuda in [False, True]:
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    if cuda:
        model.cuda()
    else:
        model.cpu()

    for ii, (inputs, labels) in enumerate(train_loader):
        inputs, labels = Variable(inputs), Variable(labels)

        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model.forward()
        loss = criterion(outputs)
        if ii == 3:
            break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNEt Model
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# Define new Classifier
classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(512, 2),
                           nn.LogSoftmax(dim=1))

model.fc = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for images, labels in train_loader:
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        model.eval()
        accuracy = 0
        test_loss = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch + 1}/{epochs}..."
              f" Train loss: {running_loss/print_every: .3f}.. "
              f" Test loss: {test_loss/len(test_loader): .3f}.."
              f" Test accuracy: {accuracy/len(test_loader): .3f}..")

        running_loss = 0
        model.train()
