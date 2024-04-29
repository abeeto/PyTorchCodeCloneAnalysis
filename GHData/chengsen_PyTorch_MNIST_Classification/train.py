import torch.nn.functional as F
import torch

__build__ = 2018
__author__ = "singsam_jam@126.com"


def train(args, model, device, train_loader, test_loader, optimizer):
    for epoch in range(1, args.epochs + 1):
        print(f"Train Epoch: {epoch}")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            correct = accuary(output, target)
            if batch_idx % args.log_interval == 0:
                print(f'[{batch_idx * len(data)}/{len(train_loader.dataset)} | '
                      f'({100. * batch_idx / len(train_loader):.0f}%)] | Loss: {loss.item():.3f} '
                      f'| correct:{correct/100:.3f}')

        test(model, device, test_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            correct += accuary(output, target)

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


def accuary(output, target):
    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct
