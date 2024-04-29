import torch


def test_net(net, criterion, test_loader, device):
    # тестирование
    test_loss = 0
    correct = 0
    need_resize = net.need_resize
    losses_count = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            if need_resize:
                data = data.view(-1, 28 * 28)

            net_out = net(data)
            # Суммируем потери со всех партий
            test_loss += criterion(net_out, labels).data
            losses_count += 1
            # получаем индекс максимального значения
            predicted = net_out.data.max(1)[1]
            # сравниваем с целевыми данными, если совпадает добавляем в correct
            correct += predicted.eq(labels.data).sum()

    test_loss /= losses_count
    test_acc = float(100. * correct / len(test_loader.dataset))
    print(f'Test set: Average loss: {test_loss:.4f},'
          f' Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({test_acc:.2f}%)')
    return test_acc
