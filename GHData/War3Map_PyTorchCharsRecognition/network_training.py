import torch


def train_net(net, train_loader, optimizer,
              criterion, epochs, device, losses_info,
              acc_info, train_batch_size):
    # запускаем главный тренировочный цикл
    # пройдёмся по батчам из наших тестовых наборов
    # каждый проход меняется эпоха
    avg_loss = 0
    correct = 0
    total = 0
    maxbatch_count = len(train_loader.dataset) // train_batch_size
    need_resize = net.need_resize

    for epoch in range(epochs):
        final_loss = 0
        correct = 0
        batch_count = 0
        for batch_id, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            # # изменим размер с (batch_size, 1, 28, 28) на (batch_size, 28*28)

            if need_resize:
                data = data.view(-1, 28*28)

            # print(f"Input shape: {data.shape}")
            # оптимизатор
            # для начала обнулим градиенты перед работой
            optimizer.zero_grad()
            # считаем выход нейросети
            net_out = net(data)
            # оптимизировать функцию потерь будем на основе целевых данных и выхода нейросети
            loss = criterion(net_out, labels)
            # делаем обратный ход
            loss.backward()
            # оптимизируем функцию потерь, запуская её по шагам полученным на backward этапе
            optimizer.step()
            # вывод информации
            print('Train Epoch: {} [{}/{} ({:.0f}%)]; Loss: {:.6f}'.format(
                epoch + 1, (batch_id + 1) * len(data), len(train_loader.dataset),
                100. * (batch_id + 1) / len(train_loader), loss.data))
            # len(train_loader.dataset)
            batch_count += 1
            final_loss += loss.data.item()
            # Отслеживание точности
            if (batch_id + 1) * len(data) == maxbatch_count * len(data):
                total = labels.size(0)
                _, predicted = torch.max(net_out.data, 1)
                correct += (predicted == labels).sum().item()
        # вносим текущее значение функции потерь
        avg_loss += final_loss / batch_count
        losses_info.append(final_loss / batch_count)
        print("Average Loss={}".format(final_loss / batch_count))
        # вносим текущее значение точности распознавания
        acc_info.append(correct / total)
        print("Accuracy={}".format(correct / total))

    # avg_loss /= len(train_loader.dataset)
    train_acc = float(100. * correct / len(train_loader.dataset))
    return train_acc
