import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from net import Net
'''
from torchvision import models
net = models.vgg16(pretrained=True)
net.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))
'''

net = Net()
net.cuda()

print(net)


loss_func = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

path = "/home/yi/Workspace/Datasets/cat_dog"

train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(path, transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
        ])),
        batch_size=32, shuffle=True,
        num_workers=0, pin_memory=True)

'''
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=False),
        batch_size=16, shuffle=True,
        num_workers=0, pin_memory=True)
'''

total_step = 0
for epoch in range(50):
    net.train()
    for idx, (images, labels) in enumerate(train_loader):
        total_step += 1

        images = images.cuda()
        labels = labels.cuda()

        predict = net(images)
        loss = loss_func(predict, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        _, predicted = torch.max(predict.data, 1)
        total = len(labels)
        correct = 0
        for i in range(total):
            if predicted.data[i] == labels.data[i]:
                correct += 1
        print("Loss: %.5f Acc: %.3f" % (loss, correct/total))

        if total_step >= 500 and total_step % 100 == 0:
            images = images.cpu()
            for i in range(len(images)):
                img = torchvision.transforms.ToPILImage()(images[i])
                plt.subplot(4, 8, i + 1)
                tit = "lab: {} pdt: {}".format(labels.data[i],
                                               predicted.data[i])
                plt.title(tit)
                plt.imshow(img)
            plt.show()




'''
for batch_id, (images, labels) in enumerate(load_dataset(path)):

    print(images.size())
    print(labels.size())
    predict = images
    loss = loss_func(predict, labels)
    
    if step % 10 == 0:

        predict_item = predict.clone()
        predict_item = predict_item.cpu().detach().numpy()
        acc = sum([int(labels_np[i] == np.where(predict_item[i] == np.amax(predict_item[i]))) for i in range(len(labels_np))])/len(labels_np)
        images = images.cpu()
        if step % 500 == 0:
            for i in range(len(images_np)):
                img = torchvision.transforms.ToPILImage()(images[i])
                plt.subplot(4, 4, i + 1)
                tit = "lab: {} pdt: {}".format(labels_np[i].item(), np.where(predict_item[i] == np.amax(predict_item[i]))[0][0])
                plt.title(tit)
                plt.imshow(img)
            plt.text(0.6, 0.6, "Step: %d" % step, size=20, ha="center", va="center",
                     bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8), ))
            plt.show()
        print("step:", step, "acc: %.3f" % acc, "loss: %f" % loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(net.state_dict(), "./model.pb")
'''
