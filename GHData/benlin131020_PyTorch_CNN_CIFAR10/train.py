import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 5
NUM_CLASSES = 10
CHANNELS = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True)
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class Cnn(nn.Module):
    def __init__(self, in_channels, n_class):  
        super(Cnn, self).__init__()
        self.conv = nn.Sequential( 
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1),  # input:3*32*32
            nn.ReLU(True), # output:16*32*32
            nn.MaxPool2d(2), # output:16*16*16
            nn.Conv2d(16, 32, 5, stride=1, padding=0),  # input:16*16*16
            nn.ReLU(True), # output:32*12*12
            nn.MaxPool2d(2))  # output:32*6*6

        self.fc = nn.Sequential(
            nn.Linear(1152, 128),  # 1152=32*6*6
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = Cnn(CHANNELS, NUM_CLASSES).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        '''
        Since PyTorch 0.4.0, Tensors and Variables have merged.
        Variable wrapping continues to work as before but returns an object of type torch.Tensor.
        img = Variable(img).to(device)
        label = Variable(label).to(device)
        '''
        img = img.to(device)
        label = label.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        # loss.data.item():average of batches
        # total loss = loss.data.item()*batch_size
        running_loss += loss.item() * label.size(0)  
        #prediction and accuracy
        _, pred = torch.max(out.data, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.item()
    # print statistics every epoch
    print('Train Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1,
        running_loss / (len(train_dataset)),
        running_acc / (len(train_dataset))))

# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
model.eval()
eval_loss = 0
eval_acc = 0
# torch.no_grad() impacts the autograd engine and deactivate it. 
# It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).
with torch.no_grad():
    for data in test_loader:
        img, label = data
        '''
        Volatile is recommended for purely inference mode, when you’re sure you won’t be even calling .backward(). 
        It will use the absolute minimal amount of memory to evaluate the model.
        Since PyTorch 0.4.0, Volatile flag have been deprecated. 
        This has now been replaced by a set of more flexible context managers including torch.no_grad(), torch.set_grad_enabled(grad_mode), and others.
        img = Variable(img, volatile=True).to(device)
        label = Variable(label, volatile=True).to(device)
        '''
        img = img.to(device)
        label = label.to(device)
        out = model(img) 
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out.data, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc * 1.0 / (len(test_dataset))))

torch.save(model.state_dict(), './cnn.pth')
