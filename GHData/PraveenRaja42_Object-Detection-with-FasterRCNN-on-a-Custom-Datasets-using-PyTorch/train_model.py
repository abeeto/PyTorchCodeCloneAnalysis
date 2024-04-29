from fastcore.test import test
import torch
from torch.utils import data
from torch_snippets import *
import data_preprocessing
import model
import os
import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter

DEFAULT_IMG_DIR = "/users/Object_detection_project/images"
DEFAULT_DF_DIR = "/users/Object_detection_project/df.csv"
N_EPOCHS = 5


def train_batch(inputs, model, optimizer, device):
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses


@torch.no_grad()
def validate_batch(inputs, model, optimizer, device):
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable cuda"
    )
    parser.add_argument(
        "--img_dir", default=DEFAULT_IMG_DIR, help="dir to images folder"
    )
    parser.add_argument(
        "--df_dr", default=DEFAULT_DF_DIR, help="dir to meta_data or label folder"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=N_EPOCHS, help="number of epochs"
    )
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    train_loader, test_loader = data_preprocessing.data_loader(args.df_dr, args.img_dir)
    runs_path = os.path.join("runs", args.run)
    os.makedirs(runs_path, exist_ok=True)
    writer = SummaryWriter(runs_path)
    model = model.get_model().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    n_epochs = args.n_epochs
    saves_path = os.path.join("saves", args.run)
    os.makedirs(saves_path, exist_ok=True)
    log = Report(n_epochs)

    for epoch in range(n_epochs):
        _n = len(train_loader)
        for ix, inputs in enumerate(train_loader):
            loss, losses = train_batch(inputs, model, optimizer, device)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [
                losses[k]
                for k in [
                    "loss_classifier",
                    "loss_box_reg",
                    "loss_objectness",
                    "loss_rpn_box_reg",
                ]
            ]
            pos = epoch + (ix + 1) / _n
            writer.add_scalar("train_loss", loss.item())
            log.record(
                pos,
                trn_loss=loss.item(),
                trn_loc_loss=loc_loss.item(),
                trn_regr_loss=regr_loss.item(),
                trn_objectness=loss_objectness.item(),
                trn_rpn_box_reg_loss=loss_rpn_box_reg.item(),
                end="\r",
            )

        _n = len(test_loader)
        for ix, inputs in enumerate(test_loader):
            loss, losses = validate_batch(inputs, model, optimizer, device)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [
                losses[k]
                for k in [
                    "loss_classifier",
                    "loss_box_reg",
                    "loss_objectness",
                    "loss_rpn_box_reg",
                ]
            ]
            pos = epoch + (ix + 1) / _n
            writer.add_scalar("val_loss", loss.item())
            log.record(
                pos,
                val_loss=loss.item(),
                val_loc_loss=loc_loss.item(),
                val_regr_loss=regr_loss.item(),
                val_objectness_loss=loss_objectness.item(),
                val_rpn_box_reg_loss=loss_rpn_box_reg.item(),
                end="\r",
            )
        if (epoch + 1) % (n_epochs // 5) == 0:
            log.report_avgs(epoch + 1)

        torch.save(
            model.state_dict(), os.path.join(saves_path, f"train_loss {loss.item()}")
        )
