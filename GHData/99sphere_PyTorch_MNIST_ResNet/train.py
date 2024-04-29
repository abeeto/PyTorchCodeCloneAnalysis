import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from model.ResNet18 import ResNet18
from model.ResNet34 import ResNet34
from model.ResNet50 import ResNet50
from model.ResNet101 import ResNet101
from model.ResNet152 import ResNet152


from utils.util import AverageMeter

BATCH_SIZE = 32
EPOCH = 1

if __name__ == "__main__":
    print("Data Loading Start")

    mnist_train_dataset = datasets.MNIST(
        root="./data/", train=True, transform=transforms.ToTensor(), download=True
    )
    mnist_test_dataset = datasets.MNIST(
        root="./data/", train=False, transform=transforms.ToTensor(), download=True
    )

    mnist_train_dataloader = DataLoader(
        mnist_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )
    mnist_test_dataloader = DataLoader(
        mnist_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )

    print("Data Loading Done")

    # res18 = ResNet18(3, 10)
    # res34 = ResNet34(3, 10)
    res50 = ResNet50(3, 10)
    # res101 = ResNet101(3, 10)
    # res152 = ResNet152(3, 10)

    # ref_res18 = models.resnet18(pretrained=False)
    # ref_res18.fc = nn.Linear(512, 10)
    # ref_res34 = models.resnet34(pretrained=False)
    # ref_res34.fc = nn.Linear(512, 10)
    # ref_res50 = models.resnet50(pretrained=False)
    # ref_res50.fc = nn.Linear(2048, 10)
    # ref_res101 = models.resnet101(pretrained=False)
    # ref_res101.fc = nn.Linear(2048, 10)
    # ref_res152 = models.resnet152(pretrained=False)
    # ref_res152.fc = nn.Linear(2048, 10)

    model = res50
    criterion = nn.CrossEntropyLoss()
    optm = optim.Adam(model.parameters(), lr=1e-3)

    log_dir = "./log"

    os.makedirs(log_dir, exist_ok=True)

    model = model.cuda()

    with open(os.path.join(log_dir, str(model) + "_train_log.csv"), "w") as log:
        model.train()
        for epoch in range(EPOCH):
            for iter, (img, label) in enumerate(mnist_train_dataloader):
                batch, channel, width, height = img.shape
                img = img.expand(batch, 3, width, height)
                img, label = img.float().cuda(), label.long().cuda()

                optm.zero_grad()

                pred_logit = model(img)

                loss = criterion(pred_logit, label)
                loss.backward()
                optm.step()

                pred_label = torch.argmax(pred_logit, 1)
                acc = (pred_label == label).sum().item() / len(img)

                train_loss = loss.item()
                train_acc = acc

                if (iter % 20 == 0) or (iter == len(mnist_train_dataloader) - 1):
                    model.eval()
                    valid_loss, valid_acc = AverageMeter(), AverageMeter()

                    for img, label in mnist_test_dataloader:
                        batch, channel, width, height = img.shape
                        img = img.expand(batch, 3, width, height)
                        img, label = img.float().cuda(), label.long().cuda()

                        with torch.no_grad():
                            pred_logit = model(img)

                        loss = criterion(pred_logit, label)

                        pred_label = torch.argmax(pred_logit, 1)
                        acc = (pred_label == label).sum().item() / len(img)

                        valid_loss.update(loss.item(), len(img))
                        valid_acc.update(acc, len(img))

                    valid_loss = valid_loss.avg
                    valid_acc = valid_acc.avg

                print(
                    "Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f | Valid Loss %.4f | Valid Acc %.4f"
                    % (
                        epoch * len(mnist_train_dataloader) + iter,
                        EPOCH * len(mnist_train_dataloader),
                        train_loss,
                        train_acc,
                        valid_loss,
                        valid_acc,
                    )
                )

                log.write(
                    "%d,%.4f,%.4f,%.4f,%.4f\n"
                    % (
                        epoch * len(mnist_train_dataloader) + iter,
                        train_loss,
                        train_acc,
                        valid_loss,
                        valid_acc,
                    )
                )
