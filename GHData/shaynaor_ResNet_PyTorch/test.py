import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from model import resnet18
from utils import parse_test_args, calc_accuracy
from dataset import Dataset

import torch


def test(test_loader, model, criterion, device, plot_flag):
    model.eval()
    epoch_loss_lst = list()

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        # Forward pass and record loss
        pred = model(x)
        loss = criterion(pred, y)
        epoch_loss_lst.append(loss.item())

        if plot_flag:
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(x.squeeze(0).permute(1, 2, 0), cmap='gray_r')
            probs = F.softmax(pred, dim=1)
            title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.2f}%)'
            plt.title(title, fontsize=8)
            fig.suptitle('Predictions')
            plt.show()

    epoch_loss = sum(epoch_loss_lst) / len(epoch_loss_lst)
    return epoch_loss


def test_loop(test_loader, model, criterion, device, plot_flag):
    test_losses = list()

    with torch.no_grad():
        test_loss = test(test_loader, model, criterion, device, plot_flag)
        test_losses.append(test_loss)

    test_acc = calc_accuracy(model, test_loader, device)

    print(f"Test loss:{test_loss:.5f} | Test accuracy:{test_acc:.5f}")


def main():
    args = parse_test_args()

    torch.manual_seed(args.manual_seed)

    model = resnet18(args.num_classes).to(args.device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss()

    dataset = Dataset(args.manual_seed, args.device)
    test_loader = dataset.get_test_loader(args.batch_size)

    test_loop(test_loader, model, criterion, args.device, args.plot_flag)


if __name__ == '__main__':
    main()
