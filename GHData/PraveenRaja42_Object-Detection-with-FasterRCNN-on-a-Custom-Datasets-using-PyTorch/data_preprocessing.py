import os
from PIL import Image
import pandas as pd

from torch_snippets import *
import glob
from sklearn.model_selection import train_test_split

# def label_ind(df):
#     label2target = {l:t+1 for t, l in enumerate(df['LabelName'].unique())}
#     label2target['backround'] = 0
#     num_classes = len(label2target)
#     return num_classes, label2target


def preprocess_img(img):
    img = torch.tensor(img).permute(2, 0, 1)
    return img.float()


class OpenDataset(torch.utils.data.Dataset):
    w, h = 224, 224

    def __init__(self, df, image_dir):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir + "/*")
        self.df = df
        self.image_infos = df.ImageID.unique()

    def __getitem__(self, ix):
        label2target = {l: t + 1 for t, l in enumerate(self.df["LabelName"].unique())}
        label2target["background"] = 0
        num_classes = len(label2target)
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR)) / 255
        data = self.df[self.df["ImageID"] == image_id]
        labels = data["LabelName"].values.tolist()
        data = data[["XMin", "YMin", "XMax", "YMax"]].values
        data[:, [0, 2]] *= self.w  # convert to absolute coordinates
        data[:, [1, 3]] *= self.h
        boxes = data.astype(np.uint32).tolist()
        # torch FRCNN expects ground truths as a dictionary of tensors
        target = {}
        target["boxes"] = torch.Tensor(boxes).float()
        target["labels"] = torch.Tensor([label2target[i] for i in labels]).long()
        img = preprocess_img(img)
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.image_infos)


def data_loader(df, img_dir):
    df = pd.read_csv(df)
    trn_ids, val_ids = train_test_split(
        df.ImageID.unique(), test_size=0.1, random_state=99
    )
    trn_df, val_df = df[df["ImageID"].isin(trn_ids)], df[df["ImageID"].isin(val_ids)]
    train_ds = OpenDataset(trn_df, img_dir)
    test_ds = OpenDataset(val_df, img_dir)
    train_loader = DataLoader(
        train_ds, batch_size=4, collate_fn=train_ds.collate_fn, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=4, collate_fn=test_ds.collate_fn, drop_last=True
    )

    return train_loader, test_loader