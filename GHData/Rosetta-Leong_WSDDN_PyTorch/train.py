import argparse
import os

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import WSDDN_Dataset
from model import WSDDN
from utils import BASE_DIR, set_seed

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Train WSDDN model")
    parser.add_argument(
        "--base_net", type=str, default="vgg", help="Base network to use"
    )
    parser.add_argument("--seed", type=int, default=61, help="Seed to use")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Epoch count")
    parser.add_argument("--offset", type=int, default=0, help="Offset count")
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard info')
    parser.add_argument(
        "--state_period", type=int, default=5, help="State saving period"
    )
    args = parser.parse_args()
    return args


def train(args):
    # Set the hyperparameters
    LR = args.lr
    WD = args.wd
    EPOCHS = args.epochs
    OFFSET = args.offset
    STATE_PERIOD = args.state_period

    # Create dataset and data loader
    train_ds = WSDDN_Dataset("trainval")  # len = 5011
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=16)

    # Create the network
    net = WSDDN(base_net=args.base_net)

    if OFFSET != 0:
        # 非首次训练，加载先前模型checkpoint
        state_path = os.path.join(BASE_DIR, "weight", f"epoch_{OFFSET}.pt")
        net.load_state_dict(torch.load(state_path))
        tqdm.write(f"Loaded epoch {OFFSET}'s state.")
    else:
        # 首次训练，加载预训练权重
        if(net.base_net == "alexnet" ):
            state_path = os.path.join(BASE_DIR, "weight", "alexnet-owt-4df8aa71.pth")
        else:
            state_path = os.path.join(BASE_DIR, "weight", "vgg16-397923af.pth")
        net.base.load_state_dict(torch.load(state_path))


    net.to(DEVICE)
    net.train()

    # Set loss function and optimizer
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    # lr = 1e-5     if epoch < 10
    # lr = 1e-6    if 10 <= epoch < 20
    # lr = 1e-7   if epoch >= 20 (which is not allowed in this project)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    scheduler.last_epoch = OFFSET

    # Visualization
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    board = SummaryWriter(log_dir=args.tensorboard_dir)

    # Train the model
    for epoch in tqdm(range(OFFSET + 1, EPOCHS + 1), "Total"):
        #epoch:1,2,...,20
        epoch_loss = 0.0

        for (
                batch_img_ids,  #图片文件名
                batch_imgs, #图片
                batch_boxes,    #EB生成的Bounding Box
                batch_scores,   #每个Box对应的得分
                batch_target,   #图片对应的类别标签labels [1,0,0,0,1...]由于有20个类对应20项
        ) in tqdm(train_dl, f"Epoch {epoch}"):
            optimizer.zero_grad()

            batch_imgs, batch_boxes, batch_scores, batch_target = (
                batch_imgs.to(DEVICE),
                batch_boxes.to(DEVICE),
                batch_scores.to(DEVICE),
                batch_target.to(DEVICE),
            )
            combined_scores = net(batch_imgs, batch_boxes, batch_scores)

            loss = WSDDN.calculate_loss(combined_scores, batch_target[0])
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()


        if epoch % STATE_PERIOD == 0:
            path = os.path.join(BASE_DIR, "weight", f"{net.base_net}_epoch_{epoch}.pt")
            torch.save(net.state_dict(), path)
            tqdm.write(f"State saved to {path}")

        tqdm.write(f"Avg loss is {epoch_loss / len(train_ds)}")
        board.add_scalar('Train_loss', epoch_loss / len(train_ds), epoch)

        scheduler.step()

    print('Finished Training')
    board.close()
    torch.save(net.state_dict(), os.path.join(BASE_DIR, "weight", "wsddn_final.pt"))



if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    train(args)

