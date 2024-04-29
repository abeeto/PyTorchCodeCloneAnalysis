import numpy as np
import pandas as pd
import cv2
import json
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self, files, label_encoder, labels=None, mode="train"
    ):
        self.mode = mode
        self.files = files
        self.max_len = 24 * 6
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split("/")[-1]

        # image
        image_path = f"{file}/{file_name}.jpg"
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255
        img = np.transpose(img, (2, 0, 1))

        if self.mode == "train":
            json_path = f"{file}/{file_name}.json"
            with open(json_path, "r") as f:
                json_file = json.load(f)

            crop = json_file["annotations"]["crop"]
            disease = json_file["annotations"]["disease"]
            risk = json_file["annotations"]["risk"]
            label = f"{crop}_{disease}_{risk}"

            return {
                "img": torch.tensor(img, dtype=torch.float32),
                "label": torch.tensor(self.label_encoder[label], dtype=torch.long),
            }
        else:
            return {
                "img": torch.tensor(img, dtype=torch.float32),
            }


class CustomDataset_ens(Dataset):
    def __init__(
        self, files, csv_feature_dict, label_encoder, labels=None, mode="train"
    ):
        self.mode = mode
        self.files = files
        self.csv_feature_dict = csv_feature_dict
        self.csv_feature_check = [0] * len(self.files)
        self.csv_features = [None] * len(self.files)
        self.max_len = 24 * 6
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split("/")[-1]

        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f"{file}/{file_name}.csv"
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace("-", 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (
                    self.csv_feature_dict[col][1] - self.csv_feature_dict[col][0]
                )
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]

        # image
        image_path = f"{file}/{file_name}.jpg"
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255
        img = np.transpose(img, (2, 0, 1))

        if self.mode == "train":
            json_path = f"{file}/{file_name}.json"
            with open(json_path, "r") as f:
                json_file = json.load(f)

            crop = json_file["annotations"]["crop"]
            disease = json_file["annotations"]["disease"]
            risk = json_file["annotations"]["risk"]
            # label = f"{crop}_{disease}_{risk}"

            return {
                "img": torch.tensor(img, dtype=torch.float32),
                "label_crop": torch.tensor(
                    self.label_encoder[0][f"{crop}"], dtype=torch.long
                ),
                "label_disease": torch.tensor(
                    self.label_encoder[1][f"{disease}"], dtype=torch.long
                ),
                "label_risk": torch.tensor(
                    self.label_encoder[2][f"{risk}"], dtype=torch.long
                ),
            }
        else:
            return {
                "img": torch.tensor(img, dtype=torch.float32),
                "csv_feature": torch.tensor(csv_feature, dtype=torch.float32),
            }
