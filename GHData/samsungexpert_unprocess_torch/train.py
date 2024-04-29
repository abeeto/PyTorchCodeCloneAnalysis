import argparse
import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset
from models import DemosaicNet
from utils import init_weight, ImagePool, LossDisplayer, rgb_augment_quad


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset_path", type=str, default="/data/team19/mit/images/train/")
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--lambda_ide", type=float, default=10)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--pool_size", type=int, default=50)
parser.add_argument("--identity", action="store_true")

args = parser.parse_args()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device="cpu"
    print(device)

    model = DemosaicNet().to(device)


    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
    else:
        print("initial")
        model.apply(init_weight)
        epoch = 0


    model.train()


    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataloader = DataLoader(
        Dataset(args.dataset_path,transform),
        batch_size = 4,
        #shuffle=True
    )
    dataset_name = os.path.basename(args.dataset_path)

    criterion_MSE  = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    disp = LossDisplayer(["loss_net"])
    #summary = SummaryWriter()

    optim_net = optim.Adam(model.parameters(), lr= args.lr)

    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler_net = optim.lr_scheduler.LambdaLR(optimizer=optim_net, lr_lambda=lr_lambda)

    os.makedirs(f"checkpoint/{dataset_name}", exist_ok=True)

    print("num data", len(dataloader))

    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")

        for idx, (img) in enumerate(dataloader):
            print(f"{idx}/{len(dataloader)}", end="\r")

            rgb_in = img.to(device)

            image_in, ref_image = rgb_augment_quad(rgb_in)

            img_out = model(image_in)

            loss = criterion_MSE(img_out, ref_image) + 0.01*criterion_L1(img_out, ref_image)

            optim_net.zero_grad()
            loss.backward()
            optim_net.step()

            disp.record([loss])

        scheduler_net.step()
        avg_losses = disp.get_avg_losses()

        #summary.add_scalar("loss", avg_losses[0], epoch)

        disp.display()
        disp.reset()

        if epoch % 10 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                },
                os.path.join("checkpoint", dataset_name, f"{epoch}.pth"),
            )

if __name__ == "__main__":
    train()





