import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from model import ConvNN
from utils import get_accuracy, plot_result
import time
import copy

# At least one transformation
transform = transforms.Compose([
    transforms.ToTensor()
])
trainset = MNIST('MNIST', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
testset = MNIST('MNIST', train = False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=True, num_workers=2)

Debug_flag = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch = 50
model = ConvNN()
model = model.to(device)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma= 0.1)

since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = []
total_acc = 0.0


'''
    Training Process
'''
for i in range(epoch):
    exp_lr_scheduler.step()
    loss_show = 0
    model.train()
    for j, train_data in enumerate(trainloader):
        optimizer.zero_grad()
        # if Debug_flag == 10:
        #     break
        train_img = train_data[0].to(device)
        label = train_data[1].to(device)

        pred = model(train_img)

        loss = loss_fn(pred, label)
        loss_show += loss
        loss.backward()
        optimizer.step()

        # Debug_flag += 1

    print("The Train loss is {0:8.5f}".format(loss_show))
    print("-" * 20)
    Debug_flag = 0

    '''
        After one epoch using testset to get the final acc
    '''
    model.eval()
    with torch.no_grad():
        for j, test_data in enumerate(testloader):
            # if Debug_flag == 2:
            #     break
            test_img = test_data[0].to(device)
            label = test_data[1].to(device)

            _, pred = model(test_img, apply_softmax=True).max(dim=1)
            # print("The prediction is ")
            # print(pred)
            # print("The label is ")
            # print(label)
            acc = get_accuracy(pred, label)
            total_acc = (acc + total_acc) / 2 if total_acc != 0.0 else acc

            # Debug_flag += 1
    print("acc is {0:4.1f}%".format(total_acc))
    best_acc.append(total_acc)
    print("{0:2d} epochs ends".format(i))

plot_result(best_acc)
torch.save(model.state_dict(), "model.pt")

