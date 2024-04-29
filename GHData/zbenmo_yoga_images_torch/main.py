import pathlib
import pandas as pd
import numpy as np
import albumentations
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from classification_dataset import (
    ImagesDataset,
    TorchDataset,
    AugmentedDataset,
    ClassificationDataset
)
from model import get_model
import training


def create_dataset(paths_of_images, augmentations, targets=None):
    ds = ImagesDataset(paths_of_images)
    ds = AugmentedDataset(ds, augmentations)
    ds = TorchDataset(ds)
    if targets is not None:
        return ClassificationDataset(ds, targets)
    else:
        return ds


def create_kfolds(df):
    """
    shuffles and adds a kfold column
    """
    # df.sample(frac=1).reset_index(drop=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=1)
    df['kfold'] = -1
    for fold, (tr_, ev_) in enumerate(skf.split(df, y=df.class_6)):
        df.loc[ev_, 'kfold'] = fold
    return df


MODEL_PATH = "model.bin" 
data_path = pathlib.Path(r'C:\Users\zbenm\projects\for_Kaggle_days_1\data')


def main():
    path = data_path / r'images\train_images'
    df = pd.read_csv(data_path / 'train.csv')
    df = create_kfolds(df)
    train_df = df.loc[lambda d: d.kfold != 0].reset_index(drop=True)
    valid_df = df.loc[lambda d: d.kfold == 0].reset_index(drop=True)
    paths_of_train_images = train_df.image_id.apply(lambda x: path / x).values
    train_targets = train_df.class_6.values
    # these values are related to ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            ),
            albumentations.Resize(227, 227, always_apply=True),
            albumentations.HorizontalFlip(p=0.5)
        ]
    ) 
    train_ds = create_dataset(paths_of_train_images, train_aug, train_targets)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers = 0 # num_worker = 0 is for CPU, simple, 1 core
    )

    paths_of_valid_images = valid_df.image_id.apply(lambda x: path / x).values
    valid_targets = valid_df.class_6.values

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            ),
            albumentations.Resize(227, 227, always_apply=True),
        ]
    ) 

    valid_ds = create_dataset(paths_of_valid_images, valid_aug, valid_targets)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=16, shuffle=False, num_workers = 0 # num_worker = 0 is for CPU, simple, 1 core
    )

    model = get_model(pretrained=True)
    device = "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 1

    for epoch in range(epochs):
        training.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = training.evaluate(
            valid_loader, model, device=device
        )
        accuracy = metrics.accuracy_score(valid_targets, predictions)
        print(f'{epoch=}, {accuracy=}')

    torch.save(model.state_dict(), MODEL_PATH)


def predict(model_path, data_path):
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    df_test = pd.read_csv(data_path / "sample_submission.csv")

    path_to_test_images = data_path / r"images\test_images"

    paths_to_test_images = df_test.image_id.apply(lambda image_id: path_to_test_images / image_id).values

    # it is the same as the valid_aug above
    # these values are related to ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            ),
            albumentations.Resize(227, 227, always_apply=True),
        ]
    ) 

    test_aug = valid_aug
    test_dataset = create_dataset(paths_to_test_images, test_aug)

    pred = (model(x['image'].unsqueeze(0)).detach().numpy() for x in test_dataset)
    pred_list=list(pred)
    predictions_raw = np.vstack(pred_list)
    predictions = np.argmax(predictions_raw, axis=1)
    df_test.class_6 = predictions
    df_test.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    # main()
    predict(MODEL_PATH, data_path)