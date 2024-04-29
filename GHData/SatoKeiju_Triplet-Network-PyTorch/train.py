import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchsummary import summary

import matplotlib.pyplot as plt

from datasets import *
from losses import *
from models import *
from parameters import args


def train(args, model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        optimizer.zero_grad()
        anc_embedding = model(anchor)
        pos_embedding = model(positive)
        neg_embedding = model(negative)
        loss = criterion(anc_embedding, pos_embedding, neg_embedding)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print(f'epoch{epoch}, batch{batch_idx+1} loss: {running_loss / 10}')
            train_loss = running_loss / 10
            running_loss = 0.

    return train_loss


if __name__ == '__main__':
    train_dic = make_datapath_dic('train')
    transform = ImageTransform(64)
    train_dataset = TripletDataset(train_dic, transform=transform, phase='train')
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TripletNet().to(device)
    criterion = TripletLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.epochs//2, (args.epochs//4)*3], gamma=0.1)

    summary(model, (3, 64, 64))
    torch.autograd.set_detect_anomaly(True)

    x_epoch_data = []
    y_train_loss_data = []

    for epoch in range(1, args.epochs+1):
        train_loss_per_epoch = train(
            args, model, train_dataloader, criterion, optimizer, epoch
        )
        scheduler.step()

        x_epoch_data.append(epoch)
        y_train_loss_data.append(train_loss_per_epoch)

    plt.plot(x_epoch_data, y_train_loss_data, color='blue', label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('loss')
    plt.show()

    if args.save_model:
        torch.save(model.state_dict(), 'TripletNet.pt')
        print('Saved model as TripletNet.pt')
