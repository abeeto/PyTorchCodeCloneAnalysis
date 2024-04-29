import os

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm.auto import tqdm

from .model import AutoRec
from .utils.data import AutoRecData, preprocess_data
from .utils.helper import get_metrics, load_config

cfg = load_config("./config/config.yaml")

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True


def load_data() -> pd.DataFrame:
    return pd.read_csv(
        cfg["data_path"],
        sep="::",
        header=None,
        names=["user", "item", "rating"],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: np.int32, 2: np.int32},
    )


# sourcery skip: remove-unused-enumerate
if __name__ == "__main__":
    ratings = load_data()

    train, test, num_users, num_items = preprocess_data(ratings)
    train_set = AutoRecData(data=train)
    test_set = AutoRecData(data=test)

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=cfg["params"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=len(test_set), shuffle=False, num_workers=0
    )

    model = AutoRec(
        num_users=num_users,
        num_items=num_items,
        num_hidden_units=cfg["params"]["num_hidden_units"],
    )
    model.cuda()
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["params"]["learning_rate"],
        weight_decay=cfg["params"]["reg_l2"],
    )

    best_epoch, best_rmse = 0, np.inf

    for epoch in tqdm(range(cfg["params"]["epochs"])):
        model.train()

        for input_vec in train_loader:
            input_mask = (input_vec > 0).cuda()
            input_vec = input_vec.float().cuda()

            model.zero_grad()
            reconstruction = model(input_vec)
            loss = loss_f(reconstruction * input_mask, input_vec * input_mask)
            loss.backward()
            optimizer.step()

        model.eval()
        rmse = get_metrics(model=model, train_set=train_set, test_set=test_set)

        print(f"[Epoch {epoch}]:: RMSE: {rmse:.6f}")

        if rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    print(f"Done. Best epoch {epoch}" f"best_rmse: {best_rmse:.6f}.")
