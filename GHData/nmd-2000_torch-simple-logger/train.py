import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from config import *

from models.model import Net
from utils.logger import SummaryWriter
from utils.utils import get_dataloader, get_dataset, get_transform


def train(model, epoch, trainloader, optimizer, loss_function):
    model.train()
    running_loss = 0
    for i, (input, target) in enumerate(trainloader, 0):
        # zero the gradient
        optimizer.zero_grad()

        # forward + backpropagation + step
        predict = model(input)
        loss = loss_function(predict, target)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()

    total_loss = running_loss / len(trainloader.dataset)
    logger.add_scalar('train/avg_loss', total_loss, global_step=epoch)

    return total_loss


def test(model, epoch, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(testloader):
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            predict = output.data.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).sum().item()

    test_loss /= len(testloader)
    test_accuracy = 100. * correct / len(testloader.dataset)

    scalars = {'loss': test_loss,
               'accuracy': test_accuracy, }

    logger.add_scalars(main_tag='eval', tag_scalar_dict=scalars, global_step=epoch)
    
    # save model
    model_path = 'exp/weight.pt'

    logger.save(model.state_dict(), model_path)

    return test_loss, test_accuracy


if __name__ == '__main__':
    # init wandb
    logger = SummaryWriter(log_dir="mlops-wandb-demo")

    # Download dataset artifact from Wandb

    # dataset_dir = "./data"
    dataset_dir = logger.data_path(local_path="./data/", dataset_name='mnist', version='latest')

    # get dataloader
    train_set, test_set = get_dataset(path=dataset_dir, transform=get_transform(), download=False) # turn off auto download
    trainloader, testloader = get_dataloader(train_set=train_set, test_set=test_set)

    # create model
    model = Net()

    # define optimizer and loss function
    epochs = EPOCHS
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # training
    pb = tqdm(range(epochs))
    train_losses, test_losses, test_accuracy = [], [], []

    # wandb watch model
    logger.watch_model(model=model, criterion=loss_function, log='all', log_freq=10)

    # training loop
    for epoch in pb:
        train_loss = train(model, epoch, trainloader, optimizer, loss_function)
        train_losses.append(train_loss)
        pb.set_description(f'Epoch: {epoch} | Train loss: {train_loss:.3f}', refresh=False)

        test_loss, test_acc = test(model, epoch, testloader)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
        pb.set_description(f'Epoch: {epoch} | Test loss: {test_loss:.3f} | Test accuracy: {test_acc:.3f}', refresh=False)
