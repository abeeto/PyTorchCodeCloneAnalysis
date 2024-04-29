import torchvision as tv

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import Net
from transforms import make_transform

if __name__ == '__main__':
    batch_size = 4
    max_epoch = 5
    # Учебный комплект
    train_set = tv.datasets.MNIST(
        root='data/',
        train=True,
        download=True,
        transform=make_transform()
    )

    train_set, validation_set = torch.utils.data.random_split(train_set, [int(0.8*len(train_set)), int(0.2*len(train_set))])

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        dataset=validation_set,
        batch_size=batch_size,
        shuffle=True
    )

    # Десять меток в наборе данных MNIST
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')

    # Создать модель сети
    model = Net()

    if torch.cuda.is_available():
        # Используйте GPU
        model.cuda()

    # Определить функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    min_valid_loss = float("inf")
    for epoch in range(max_epoch):
        running_loss = 0.0
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            # Введите данные
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs, labels = Variable(inputs), Variable(labels)

            # Градиент очистить 0
            optimizer.zero_grad()

            # forward + backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Обновить параметры
            optimizer.step()

            # Распечатать информацию журнала
            running_loss += loss.item()
            train_loss += loss.item() * len(data)

            # Печать статуса тренировки каждые 2000 партий
            if i % 1000 == 999:
                print(
                    f'[{epoch + 1}/{max_epoch}][{(i + 1) * batch_size}/{len(train_set)}] '
                    f'loss: {round(running_loss / 1000, 6)}')
                running_loss = 0.0

        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for data, labels in validation_loader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Forward Pass
            outputs = model(data)
            # Find the Loss
            loss = criterion(outputs, labels)
            # Calculate Loss
            valid_loss += loss.item() * len(data)

        mean_train_loss = train_loss / len(train_loader)
        mean_valid_loss = valid_loss / len(validation_loader)
        print(f'############################################\n'
              f'Epoch {epoch + 1} \t\t '
              f'Training Loss: {mean_train_loss} \t\t '
              f'Validation Loss: {mean_valid_loss}\n'
              f'############################################')

        if min_valid_loss > mean_valid_loss:
            print(f'Validation Loss Decreased({round(min_valid_loss, 6)}--->{round(mean_valid_loss, 6)}) '
                  f'\nSaving The Model\n'
                  f'############################################')
            min_valid_loss = mean_valid_loss

            # Saving State Dict
            torch.save(model.state_dict(), 'model.pth')
    print('Finished Training')
