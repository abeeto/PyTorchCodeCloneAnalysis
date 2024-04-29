import torch
import torch.nn as nn

from utils import calc_accuracy, parse_train_args, plot_loss_accuracy, save_checkpoint, load_checkpoint
from model import VGG16
from dataset import Dataset
from tqdm import tqdm


def train(train_loader, model, criterion, optimizer, device):
    model.train()
    epoch_loss_lst = list()

    for x, y in train_loader:
        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        epoch_loss_lst.append(loss.item())

        loss.backward()
        optimizer.step()

    epoch_loss = sum(epoch_loss_lst) / len(epoch_loss_lst)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    model.eval()
    epoch_loss_lst = list()

    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)

        # Forward pass and record loss
        pred = model(x)
        loss = criterion(pred, y)
        epoch_loss_lst.append(loss.item())

    epoch_loss = sum(epoch_loss_lst) / len(epoch_loss_lst)
    return epoch_loss


def train_loop(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device, plot_flag,
               save_checkpoint_every, model_dir, start_epoch):
    train_loss_lst, valid_loss_lst = list(), list()
    train_acc_lst, valid_acc_lst = list(), list()

    end_epoch = num_epochs + start_epoch
    for epoch in tqdm(range(start_epoch, end_epoch)):
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_loss_lst.append(train_loss)

        with torch.no_grad():
            valid_loss = validate(valid_loader, model, criterion, device)
            valid_loss_lst.append(valid_loss)

        train_acc = calc_accuracy(model, train_loader, device)
        train_acc_lst.append(train_acc)
        valid_acc = calc_accuracy(model, valid_loader, device)
        valid_acc_lst.append(valid_acc)

        print(f"Epoch:{epoch} | Train loss:{train_loss:.5f} | Validation loss:{valid_loss:.5f} | "
              f"Train accuracy:{train_acc:.5f} | Validation accuracy:{valid_acc:.5f}")

        if epoch % save_checkpoint_every == 0 and epoch != 0 and epoch != start_epoch:
            save_checkpoint(model_dir, epoch, model, optimizer, train_loss)

    save_checkpoint(model_dir, end_epoch, model, optimizer, train_loss_lst[-1])

    if plot_flag:
        plot_loss_accuracy(train_loss_lst, valid_loss_lst, train_acc_lst, valid_acc_lst)


def main():
    args = parse_train_args()

    torch.manual_seed(args.manual_seed)

    model = VGG16(args.num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0

    if args.load_model:
        start_epoch, _ = load_checkpoint(args.load_model_path, model, optimizer)

    dataset = Dataset(args.manual_seed, args.device)
    train_loader, valid_loader = dataset.get_train_valid_loaders(args.training_set_ratio, args.batch_size)

    train_loop(model, criterion, optimizer, train_loader, valid_loader, args.num_epoch, args.device, args.plot_flag,
               args.save_checkpoint, args.model_dir, start_epoch)


if __name__ == '__main__':
    main()
