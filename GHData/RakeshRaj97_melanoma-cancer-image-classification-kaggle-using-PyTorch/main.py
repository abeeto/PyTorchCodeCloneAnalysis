import os
import torch

import pretrainedmodels
import albumentations
import numpy as np
import pandas as pd

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.cuda import amp
# from apex import amp
from sklearn import metrics

from wtfml.data_loaders.image import ClassificationLoader
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        #base model
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1, 1).type_as(out)
        )
        return out, loss

def train(fold):
    training_data_path = "/train"
    model_path = "/models"
    df = pd.read_csv("train_folds.csv")
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.244, 0.255)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )

    #scaler = amp.GradScaler()

    # model, optimizer = amp.initialize(
    #     model,
    #     optimizer,
    #     opt_level="O1",
    #     verbosity=0
    # )

    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device,
            fp16=True
        )
        predictions, valid_loss = Engine.evaluate(
            valid_loader,
            model,
            device=device
        )
        predictions = np.vstack(predictions).ravel()
        #auc = metrics.roc_auc_score(valid_targets, predictions)
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        #print(f"epoch= {epoch}, AUC= {auc}")
        print(f"Epoch = {epoch}, AUC = {auc}")
        es(auc, model, os.path.join(model_path, f"model{fold}.bin"))
        if es.early_stop:
            print("early stopping")
            break


def predict(fold):
    test_data_path = ""
    model_path = ""
    df_test = pd.read_csv("")
    df_test.loc[:, "target"] = 0
    device = "cuda"
    epochs = 50
    test_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.244, 0.255)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = df_test.image_name.values.tolist()
    test_images = [os.path.join(test_data_path, i + ".jpg") for i in test_images]
    test_targets = np.zeros(len(test_images))


    test_dataset = ClassificationLoader(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(os.path.join(model_path, f"model{fold}.bin")))
    model.to(device)

    predictions = Engine.predict(
        test_loader,
        model,
        device=device
    )
    return np.vstack((predictions)).ravel()


if __name__ == "__main__":
    train(0)
    train(1)
    train(2)
    train(3)
    train(4)
    train(5)
    train(6)
    train(7)
    train(8)
    train(9)

    # p1 = predict(0)
    # p2 = predict(1)
    # p3 = predict(2)
    # p4 = predict(3)
    # p5 = predict(4)
    # p6 = predict(5)
    # p7 = predict(6)
    # p8 = predict(7)
    # p9 = predict(8)
    # p10 = predict(9)
    #
    # predictions = (p1+p2+p3+p4+p5+p6+p7+p8+p9+p10)/10
    # sample = pd.read_csv("")
    # sample.loc[:, "target"] = predictions
    # sample.to_csv("submission.csv", index=False)





