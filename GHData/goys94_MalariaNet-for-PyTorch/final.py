import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, datasets, utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time

class malariaVer1(nn.Sequential):

    def __init__(self):
        super(malariaVer1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 2)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out


#############################
"""preparing the datasets."""
#############################


data_transforms = transforms.Compose([transforms.Resize((56, 56)),
                                      transforms.ColorJitter(0.07),
                                      transforms.RandomVerticalFlip(),
                                      #transforms.RandomRotation(18),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])


img_dir = '/home/hci/soda/cell/cell_images/'
dataset = datasets.ImageFolder(img_dir, transform=data_transforms)

num_workers = 0

valid_size = 0.2
test_size = 0.1

num_train = len(dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
test_split = int(np.floor((valid_size+test_size) * num_train))
valid_split = int(np.floor((valid_size) * num_train))
valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

print(len(valid_idx), len(test_idx), len(train_idx))

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                           sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=20,
                                          sampler=test_sampler, num_workers=num_workers)


def imshow(inp, title=None):
    #########################
    """imshow for tensor."""
    #########################

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

inputs, classes = next(iter(valid_loader))
out = utils.make_grid(inputs)
imshow(out)



model = malariaVer1()

#model = models.resnet18(pretrained=True)
#model = models.resnet50(pretrained=True)
#model = models.googlenet(pretrained=True)

#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(2048, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001 , momentum=0.9)
e_list, tl_list, vl_list, tac_list, vac_list = []


def drawGraph(x, y1, y2, y3, y4):
    listX = x
    listY1 = y1
    listY2 = y2
    listY3 = y3
    listY4 = y4

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.plot(listX, listY1, '-b', label='Training Loss')
    ax1.plot(listX, listY2, '-r', label='Validation Loss')
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Loss")

    ax2.plot(listX, listY3, '-b', label='Train Acc')
    ax2.plot(listX, listY4, '-r', label='Valid Acc')
    ax2.legend(loc='lower right')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")

    plt.savefig('./output4.png')
    plt.show()


def train(n_epochs, model, optimizer, criterion, use_cuda, save_path):

    since = time.time()

    # initialize tracker
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables
        train_loss = 0.0
        valid_loss = 0.0

        correct = 0.
        total = 0.
        test_ac = 0.
        test_corrrect = 0.
        test_total = 0.
        valid_ac = 0.

        #####################
        """train the model"""
        #####################
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # initialize weights to zero
            optimizer.zero_grad()

            # forward pass
            output = model(data)

            # calculate loss
            loss = criterion(output, target)

            # back propagation
            loss.backward()

            # weight update
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            pred = output.data.max(1, keepdim=True)[1]

            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                      (epoch, batch_idx + 1, train_loss))

        test_corrrect = correct
        test_total = total
        test_ac = 100. * correct / total
        correct = 0.
        total = 0.

        ########################
        """validate the model"""
        ########################
        model.eval()

        for batch_idx, (data, target) in enumerate(valid_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            # update the average validation loss
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            pred = output.data.max(1, keepdim=True)[1]

            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss))

        print('Epoch: {} \tTrain Accuracy: %.2f%% (%2d/%2d) \tValidation Accuracy: %.2f%% (%2d/%2d)'.format(epoch) % (
            test_ac, test_corrrect, test_total, 100. * correct / total, correct, total))

        valid_ac = 100. * correct / total

        #save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss

        e_list.append(epoch)
        tl_list.append(train_loss)
        vl_list.append(valid_loss)
        tac_list.append(test_ac)
        vac_list.append(valid_ac)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model

train(10, model, optimizer, criterion, device, 'malaria_detection.pt')
drawGraph(e_list, tl_list, vl_list, tac_list, vac_list)


model.load_state_dict(torch.load('malaria_detection.pt'))
def test(model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        # forward pass
        output = model(data)

        # calculate the loss
        loss = criterion(output, target)

        # update test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %.2f%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


test(model, criterion, device)
