import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from model import Net
from dataset import get_data_loaders
from train import train


def main():
    # 定数
    epochs = 100
    train_batchsize = 128
    valid_batchsize = 4
    log_interval = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 各学習に必要なもの
    model = Net(num_class=10)
    train_loader, valid_loader = get_data_loaders(train_batchsize, valid_batchsize)
    criterion = F.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    log_writer = SummaryWriter('./log')

    # 学習開始
    train(epochs=epochs, model=model,
          train_loader=train_loader, valid_loader=valid_loader,
          criterion=criterion, optimizer=optimizer,
          writer=log_writer, device=device, log_interval=log_interval)

    # モデル保存
    torch.save(model.state_dict(), './checkpoints/final_weights.pt')

    log_writer.close()


if __name__ == '__main__':
    main()
