import torch
from torch import optim, nn
from torchvision import datasets, transforms


transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.FashionMNIST('FashionMNIST_data/', download=False, train=True, transform=transform)
test_set = datasets.FashionMNIST('FashionMNIST_data/', download=False, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten
        images = images.view(images.shape[0], -1)

        # Training Pass
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)

        # Backward
        loss.backward()

        # Update optimizer
        optimizer.step()
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.shape[0], -1)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))

        print("Epoch: {}/{}..".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
              "Test Accuracy: {:.3f}.. ".format(accuracy/len(test_loader)))
