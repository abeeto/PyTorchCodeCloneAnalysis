import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import torch
import os


def calc_accuracy(model, data_loader, device):
    correct_pred = 0
    instance_count = 0

    with torch.no_grad():
        model.eval()
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            _, pred_labels = torch.max(F.softmax(pred, dim=1), 1)

            instance_count += y.size(0)
            correct_pred += (pred_labels == y).sum().item()

    acc = correct_pred / instance_count
    return acc


def parse_train_args():
    parser = argparse.ArgumentParser(description="Training argument parser")

    parser.add_argument("-ne", "--num_epoch", type=int, required=False, default=10, help="number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, required=False, default=50, help="batch size")
    parser.add_argument("-ms", "--manual_seed", type=int, required=False, default=0, help="random seed")
    parser.add_argument("-d", "--device", type=str, required=False, default="cuda", help="device: cuda/cpu")
    parser.add_argument("-lr", "--learning_rate", type=float, required=False, default=0.001, help="learning rate")
    parser.add_argument("-nc", "--num_classes", type=int, required=False, default=10, help="number of classes")
    parser.add_argument("-tr", "--training_set_ratio", type=float, required=False, default=0.8,
                        help="the ratio between training and validation set")
    parser.add_argument("-mp", "--model_dir", type=str, required=False, default="./checkpoints",
                        help="model path to save")
    parser.add_argument("-pf", "--plot_flag", type=bool, required=False, default=True, help="plot all test results")
    parser.add_argument("-sc", "--save_checkpoint", type=int, required=False, default=10,
                        help="saving checkpoint every save_checkpoint epochs")
    parser.add_argument("-lm", "--load_model", type=bool, required=False, default=False,
                        help="load pretrained model, on restart case")
    parser.add_argument("-lmp", "--load_model_path", type=str, required=False,
                        help="path of the pretrained model to load, on restart case")

    args = parser.parse_args()
    return args


def parse_test_args():
    parser = argparse.ArgumentParser(description="Testing argument parser")

    parser.add_argument("-d", "--device", type=str, required=False, default="cpu", help="device: cuda/cpu")
    parser.add_argument("-nc", "--num_classes", type=int, required=False, default=10, help="number of classes")
    parser.add_argument("-ms", "--manual_seed", type=int, required=False, default=0, help="random seed")
    parser.add_argument("-bs", "--batch_size", type=int, required=False, default=1, help="batch size")
    parser.add_argument("-mp", "--model_path", type=str, required=True, help="model path to load")
    parser.add_argument("-pf", "--plot_flag", type=bool, required=False, default=False, help="plot all test results")

    args = parser.parse_args()
    return args


def plot_loss_accuracy(train_loss_lst: list, valid_loss_lst: list, train_acc_lst: list, valid_acc_lst: list):
    fig = plt.figure()
    plt.plot(train_loss_lst, '-bo')
    plt.plot(valid_loss_lst, '-go')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    fig.suptitle('Loss')

    fig = plt.figure()
    plt.plot(train_acc_lst, '-bo')
    plt.plot(valid_acc_lst, '-go')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'])
    fig.suptitle('Accuracy')

    plt.show()


def save_checkpoint(dir_path, epoch, model, optim, loss):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    path = f"{dir_path}/checkpoint_{epoch}.pt"

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss
                },
               path)


def load_checkpoint(path, model, optim):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

