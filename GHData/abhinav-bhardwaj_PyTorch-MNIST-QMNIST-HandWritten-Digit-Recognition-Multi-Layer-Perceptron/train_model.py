from sudokunet import SudokuNet
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch import nn, optim, save, no_grad
import torch
from torchvision import datasets, transforms
torch.manual_seed(101)
from sklearn.metrics import classification_report

LR = 1e-3
BS = 128
EPOCHS = 10
TR_SPLIT = 0.70
VAL_SPLIT = 0.30

transform = transforms.Compose(
    [transforms.CenterCrop(25), transforms.Resize(28), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,)), ])

trainset_m = datasets.MNIST(r'..\input\MNIST', download=True, train=True, transform=transform)
testset_m = datasets.MNIST(r'..\input\MNIST', download=True,
                         train=False, transform=transform)

trainset_q = datasets.QMNIST(r'..\input\QMNIST', download=True, train=True, transform=transform)
testset_q = datasets.QMNIST(r'..\input\QMNIST', download=True, train=False, transform=transform)

trainset = ConcatDataset([trainset_m, trainset_q])
testset = ConcatDataset([testset_m, testset_q])

numTrain = int(len(trainset)*TR_SPLIT)
numVal = int(len(trainset)*VAL_SPLIT)
(trainset, valset) = random_split(trainset, [numTrain, numVal], generator=torch.Generator().manual_seed(42))

trainloader = DataLoader(trainset, batch_size=BS, shuffle=True)
valloader = DataLoader(valset, batch_size=BS, shuffle=True)
testloader = DataLoader(testset, batch_size=BS, shuffle=True)

#print(testset.targets.cpu().detach().numpy())
#print(trainset.dataset.classes)

Tstep = len(trainloader.dataset)//BS
Vstep = len(valloader.dataset)//BS

model = SudokuNet()
print(model)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.RMSprop(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#image, label = next(iter(trainloader))
#print("Image Shape : ",image.shape)

train_losses = []
val_losses = []
train_acc = []
val_acc = []

for e in range(EPOCHS):
    model.train()
    train_cc = 0
    val_cc = 0
    train_ll = 0
    val_ll = 0
    for (X_train, y_train) in trainloader:
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_ll += loss
        train_cc += (y_pred.argmax(1)==y_train).type(torch.float).sum().item()
        
    scheduler.step()
    scheduler2.step()

    with no_grad():
        model.eval()
        for (X_val, y_val) in valloader:
            y_pred = model(X_val)
            val_ll += criterion(y_pred, y_val)
            val_cc += (y_pred.argmax(1)==y_val).type(torch.float).sum().item()
    
    avgTloss = train_ll/Tstep
    avgVloss = val_ll/Vstep

    train_cc = train_cc/len(trainloader.dataset)
    val_cc = val_cc/len(valloader.dataset)

    train_losses.append(avgTloss.cpu().detach().numpy())
    train_acc.append(train_cc)
    val_losses.append(avgVloss.cpu().detach().numpy())
    val_acc.append(val_cc)

    print("epoch: {}/{} ".format(e+1, EPOCHS))
    print("Train loss: {:.6f} Train accuracy: {:.6f}".format(avgTloss,train_cc))
    print("Val loss: {:.6f} Val accuracy: {:.6f}".format(avgVloss,val_cc))

with no_grad():
    model.eval()
    preds = []
    for (X_test,y_test) in testloader:
        y_pred = model(X_test)
        preds.extend(y_pred.argmax(axis=1).cpu().numpy())

save(model, 'digit_model.pt')
print("Model saved")
