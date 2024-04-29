import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd


class Model(nn.Module):
    def __init__(self, in_features, out_features, out_size_layer_1, out_size_layer_2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_size_layer_1)
        self.fc2 = nn.Linear(out_size_layer_1, out_size_layer_2)
        self.out = nn.Linear(out_size_layer_2, out_features)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = F.relu(x)
        x = self.fc2.forward(x)
        x = F.relu(x)
        out = self.out.forward(x)
        return out


if __name__ == "__main__":

    # DATASETS #########################################
    data = pd.read_csv("./PyTorchRefresher/data/iris.csv")
    features = data.drop(["target"], axis=1).values
    label = data.target.values
    # shuffle and split
    ids = np.array(data.index.unique())
    np.random.shuffle(ids)
    train_ids = ids[: int(len(data) * 0.8)]
    val_ids = ids[int(len(data) * 0.8) :]
    train_x, train_y = features[train_ids], label[train_ids]
    val_x, val_y = features[val_ids], label[val_ids]
    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)
    print(train_x.shape)
    # conv to tensor
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.long)

    # inst. model
    model = Model(4, 3, 8, 16)
    print(model.forward(train_x[0:4]))
    print(model.parameters)
    # TRAINING
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(3000):
        y_pred = model.forward(train_x)
        loss = criterion(y_pred, train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_y_pred = model.forward(val_x)
            loss_val = criterion(val_y_pred, val_y)
        if epoch % 1000 == 0:
            print(epoch, loss.item(), loss_val.item())

    ## ACCURACY
    with torch.no_grad():
        y_pred = model.forward(train_x)
        y_p = np.argmax(y_pred.detach().numpy(), axis=1)
    corr = 0
    false_ = 0
    for act, pred in zip(train_y, y_p):
        if act == pred:
            corr += 1
        else:
            false_ += 1
    print("training accuracy {}".format(corr / (corr + false_)))

    with torch.no_grad():
        y_pred = model.forward(val_x)
        y_p = np.argmax(y_pred.detach().numpy(), axis=1)
    corr = 0
    false_ = 0
    for act, pred in zip(val_y, y_p):
        if act == pred:
            corr += 1
        else:
            false_ += 1
    print("validation accuracy {}".format(corr / (corr + false_)))

    # SAVE MODEL
    torch.save(
        model.state_dict(), "PyTorchRefresher/models/iris_model.pt"
    )  # just the state dict
    torch.save(
        model, "PyTorchRefresher/models/iris_model_complete.pt"
    )  # save entire model class and state dict
