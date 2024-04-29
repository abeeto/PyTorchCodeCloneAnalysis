import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Qt5Agg', warn=False, force=True)

# Get data
data_dir = 'Cat_dog_data'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download pretrained model
model = models.resnet18(pretrained=True)

# Freze features
for param in model.parameters():
    param.requires_grad = False

# Define our new classifier
classifier = nn.Sequential(nn.Linear(512, 128),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(128,2),
                           nn.LogSoftmax(dim=1))

# Attach new classifier to model
model.fc = classifier

# Define loss
criterion = nn.NLLLoss()

# Define optimizier
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

# Activate CPU or GPU
model.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0

            for images, labels in testloader:

                images, labels = images.to(device), labels.to(device)

                logps = model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()

                # calcualte our accuracy
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                  "Test Accuracy: {:.3f}.. ".format(accuracy / len(testloader)))

            running_loss = 0
            model.train()