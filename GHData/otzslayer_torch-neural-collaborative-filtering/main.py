import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .model import NCF
from .utils.data import NCFData, get_dataset
from .utils.helper import load_config
from .utils.metrics import get_metrics

cfg = load_config("./config/config.yaml")

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=True, help="Save the model")
parser.add_argument("--gpu", type=str, default="0", help="GPU card ID")
args = parser.parse_args()

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


def construct_data():
    """Helper function to load data"""
    train_data, test_data, num_users, num_items, train_mat = get_dataset()

    # construct dataset
    train_set = NCFData(
        data=train_data,
        num_items=num_items,
        train_mat=train_mat,
        num_negative_samples=cfg["params"]["num_negative_samples"],
        is_training=True,
    )
    test_set = NCFData(
        data=test_data,
        num_items=num_items,
        train_mat=train_mat,
        num_negative_samples=0,
        is_training=False,
    )
    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=cfg["params"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=cfg["params"]["num_test_negative_samples"] + 1,
        shuffle=False,
        num_workers=0,
    )
    return num_users, num_items, train_loader, test_loader


def load_pretrain_models():
    if cfg["model_type"] != "NeuMF-pre":
        return None, None
    pretrained_gmf = torch.load(cfg["model_path"]["gmf"])
    pretrained_mlp = torch.load(cfg["model_path"]["mlp"])
    return pretrained_gmf, pretrained_mlp


# sourcery skip: remove-unused-enumerate
if __name__ == "__main__":
    num_users, num_items, train_loader, test_loader = construct_data()
    (
        num_users,
        num_items,
        pretrained_gmf,
        pretrained_mlp,
    ) = load_pretrain_models()

    model = NCF(
        num_users=num_users,
        num_items=num_items,
        num_factors=cfg["params"]["num_factors"],
        num_layers=cfg["params"]["num_layers"],
        dropout=cfg["params"]["dropout"],
        model_type=cfg["model_type"],
        pretrained_gmf=pretrained_gmf,
        pretrained_mlp=pretrained_mlp,
    )
    model.cuda()
    loss_f = nn.BCEWithLogitsLoss()

    if cfg["model_type"] == "NeuMF-pre":
        optimizer = optim.SGD(
            model.parameters(), lr=cfg["params"]["learning_rate"]
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=cfg["params"]["learning_rate"]
        )

    # metrics
    best_hr, best_ndcg, best_epoch = 0, 0, 0

    # Gradient scaler
    scaler = GradScaler()

    # training process
    for epoch in range(cfg["params"]["epochs"]):
        model.train()
        train_loader.dataset.negative_sampling()

        for idx, (user, item, label) in enumerate(tqdm(train_loader)):
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            model.zero_grad()
            with autocast():
                pred = model(user, item)
                loss = loss_f(pred, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        hr, ndcg = get_metrics(model, test_loader, cfg["params"]["top_k"])

        print(f"[Epoch {epoch}] :: Hit Ratio: {hr:.3f}\tNDCG: {ndcg:.3f}")

        if hr > best_hr:
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
            if args.out:
                if not os.path.exists(cfg["model_path"][cfg["model_type"]]):
                    os.mkdir(cfg["model_path"][cfg["model_type"]])
                torch.save(
                    model,
                    f'{cfg["model_path"][cfg["model_type"]]}{cfg["model_type"]}.pth',
                )

    print(
        f"Done. Best epoch {epoch}"
        f"Hit Ratio: {best_hr:.3f}, NDCG: {best_ndcg:.3f}."
    )
