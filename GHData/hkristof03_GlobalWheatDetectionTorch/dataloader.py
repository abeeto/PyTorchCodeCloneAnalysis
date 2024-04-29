from ast import literal_eval
from typing import Tuple, Callable
import random

from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import pandas as pd
import numpy as np

from utils.config_parser import parse_args, parse_yaml
from transformations import transforms

DIR_INPUT = './datasets'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
SEED = 2020

random.seed(2020)


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, to_tensor, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.to_tensor = to_tensor
        self.transforms = transforms

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        records = self.df.loc[(self.df['image_id'] == image_id)]

        image = Image.open(f'{self.image_dir}/{image_id}.jpg')
        image = np.array(image)

        bboxes = records[['x1','y1','x2','y2']].values

        areas = torch.as_tensor(self.df['area'].values, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = areas
        target['iscrowd'] = iscrowd

        if self.transforms:
            image, bboxes_aug = self.transforms(
                image=image,
                bounding_boxes=bboxes
            )
            target['boxes'] = bboxes_aug

        else:
            image = image.astype(np.float32)
            image /= 255.0
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.to_tensor(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(
                tuple(map(
                    lambda x: torch.tensor(x, dtype=torch.float32),
                    zip(*sample['bboxes']))
                )
            ).permute(1, 0)

        if np.all(bboxes == 0):
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([index]),
                "area": torch.zeros(0, dtype=torch.float32),
                "masks": torch.zeros((0, image.shape[0], image.shape[1]),
                    dtype=torch.uint8),
                "keypoints": torch.zeros((17, 0, 3), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        return image, target, image_id

    def __len__(self):

        return self.image_ids.shape[0]


def transform_df(path_df: str) -> pd.DataFrame:
    """Transforms initial df.
    """
    df = pd.read_csv(path_df)
    df['bbox'] = df['bbox'].apply(lambda x: literal_eval(x))
    bboxes = list(df['bbox'])
    imgaug_boxes = []

    for bbox in bboxes:
        xmin, ymin, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        imgaug_boxes.append([xmin, ymin, xmin+width, ymin+height])

    df['imgaug_bbox'] = imgaug_boxes
    box_data = np.stack(df['imgaug_bbox'])
    df[['x1','y1','x2','y2']] = pd.DataFrame(box_data).astype(np.float)
    areas = (box_data[:, 3] - box_data[:, 1]) * (box_data[:, 2] - box_data[:, 0])
    df['area'] = areas

    return df


def split_stratifiedKFolds_bbox_count(df: pd.DataFrame, n_splits: int):
    """
    """
    df_folds = df[['image_id']].copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = df[['image_id', 'source']].groupby(
        'image_id'
    ).min()['source']

    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0
    train_test_split = skf.split(X=df_folds.index, y=df_folds['stratify_group'])

    for fold_number, (train_index, val_index) in enumerate(train_test_split):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    df_folds.reset_index(inplace=True)
    df = pd.merge(df, df_folds, how='left', on='image_id')

    return df

def get_train_valid_df_skfold(df: pd.DataFrame, fold: int):
    """fold indicates the test fold."""
    train_df = df.loc[(df['fold'] != fold)].copy()
    valid_df = df.loc[(df['fold'] == fold)].copy()

    return train_df, valid_df

def get_train_valid_df(
    path_df: str,
    valid_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    """
    df = transform_df(path_df)

    image_ids = df['image_id'].unique()
    random.shuffle(image_ids)
    valid_size = round(len(image_ids) * valid_size)

    train_ids = image_ids[:-valid_size]
    valid_ids = image_ids[-valid_size:]

    train_df = df.loc[(df['image_id'].isin(train_ids))].copy()
    valid_df = df.loc[(df['image_id'].isin(valid_ids))].copy()

    return train_df, valid_df

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_valid_dataloaders(
    config_dataloader: dict,
    train_df,
    valid_df,
    collate_fn: Callable,
    train_trf: Callable = None,
    valid_trf: Callable = None,
    ) -> Tuple[DataLoader, DataLoader]:
    """"""
    config_train_loader = config_dataloader['train_loader']
    config_valid_loader = config_dataloader['valid_loader']
    dir_train = config_dataloader['train_dataset']['dir_train']

    train_dataset = WheatDataset(
        train_df,
        dir_train,
        transforms=train_trf,
        to_tensor=transforms.to_tensor(),
    )
    valid_dataset = WheatDataset(
        valid_df,
        dir_train,
        transforms=None,
        to_tensor=valid_trf,
    )
    train_data_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **config_train_loader
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        collate_fn=collate_fn,
        **config_valid_loader
    )
    return (train_data_loader, valid_data_loader)


if __name__ == '__main__':

    args = parse_args()
    config = parse_yaml(args.pyaml)
    config_dataloader = config['dataloader']

    path_df = config_dataloader['path_df']
    df = transform_df(path_df)

    train_trf = transforms.ImgAugTrainTransform()
    valid_trf = transforms.to_tensor()

    if config_dataloader['stratifiedKFold']:
        df = split_stratifiedKFolds_bbox_count(df,
            config_dataloader['n_splits'])
        folds = list(df['fold'].unique())

        for i, fold in enumerate(folds):

            train_df, valid_df = get_train_valid_df_skfold(df, fold)
            train_data_loader, valid_data_loader = get_train_valid_dataloaders(
                config_dataloader, train_df, valid_df, collate_fn,
            train_trf, valid_trf)
            print(len(train_data_loader))
            images, targets, image_ids = next(iter(train_data_loader))
    else:
        train_df, valid_df = get_train_valid_df(path_df)

        train_data_loader, valid_data_loader = get_train_valid_dataloaders(
            config_dataloader,
            collate_fn
        )
        print(len(train_data_loader))

        images, targets, image_ids = next(iter(train_data_loader))

        print(f"Length of Train dataset: {len(train_data_loader.dataset)}")
