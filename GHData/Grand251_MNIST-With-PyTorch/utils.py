import torch.nn.functional as F
import torch


def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs):

    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        loss_epoch = 0

        for batch_idx, (x, target) in enumerate(train_loader):
            y_prediction = model(x)
            loss = criterion(y_prediction, target)
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("EPOCH: " + str(epoch) + " AVG LOSS: " + str(loss_epoch / len(train_loader)))
        test_acc_history.append(test_model(model, test_loader))
        train_acc_history.append(test_model(model, train_loader))

    return train_acc_history, test_acc_history


def test_model(model, test_loader):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

    return 100. * correct / len(test_loader.dataset)

