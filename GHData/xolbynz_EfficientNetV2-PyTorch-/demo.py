import argparse
import copy
import csv
import os

import torch
import tqdm
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from glob import glob
from nets import nn
from utils import util
from CustomDataset import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm


def csv_reader(enc_dec_mode=0):
    csv_features = [
        "내부 온도 1 평균",
        "내부 온도 1 최고",
        "내부 온도 1 최저",
        "내부 습도 1 평균",
        "내부 습도 1 최고",
        "내부 습도 1 최저",
        "내부 이슬점 평균",
        "내부 이슬점 최고",
        "내부 이슬점 최저",
    ]
    csv_files = sorted(glob("/DL_data_big/DACON_2022_LeafChallenge/leaf_data/data/train/*/*.csv"))

    # temp_csv = pd.read_csv(csv_files[0])[csv_features]
    # max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # # feature 별 최대값, 최솟값 계산
    # for csv in tqdm(csv_files[1:]):
    #     temp_csv = pd.read_csv(csv)[csv_features]
    #     temp_csv = temp_csv.replace("-", np.nan).dropna()
    #     if len(temp_csv) == 0:
    #         continue
    #     temp_csv = temp_csv.astype(float)
    #     temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
    #     max_arr = np.max([max_arr, temp_max], axis=0)
    #     min_arr = np.min([min_arr, temp_min], axis=0)

    # feature 별 최대값, 최솟값 dictionary 생성
    min_arr = np.array([3.4, 3.4, 3.3, 23.7, 25.9, 0, 0.1, 0.2, 0.0])
    max_arr = np.array([46.8, 47.1, 46.6, 100, 100, 100, 34.5, 34.7, 34.4])

    csv_feature_dict = {
        csv_features[i]: [min_arr[i], max_arr[i]] for i in range(len(csv_features))
    }
    # 변수 설명 csv 파일 참조
    crop = {"1": "딸기", "2": "토마토", "3": "파프리카", "4": "오이", "5": "고추", "6": "시설포도"}
    disease = {
        "1": {
            "a1": "딸기잿빛곰팡이병",
            "a2": "딸기흰가루병",
            "b1": "냉해피해",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "2": {
            "a5": "토마토흰가루병",
            "a6": "토마토잿빛곰팡이병",
            "b2": "열과",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "3": {
            "a9": "파프리카흰가루병",
            "a10": "파프리카잘록병",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "4": {
            "a3": "오이노균병",
            "a4": "오이흰가루병",
            "b1": "냉해피해",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "5": {
            "a7": "고추탄저병",
            "a8": "고추흰가루병",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "6": {"a11": "시설포도탄저병", "a12": "시설포도노균병", "b4": "일소피해", "b5": "축과병"},
    }
    risk = {"1": "초기", "2": "중기", "3": "말기"}

    label_description = {}  # classification 111 number ex) '딸기_다량원소결핍 (P)_말기'

    label_description_crop = {}
    label_description_disease = {}
    label_description_risk = {}
    for key, value in disease.items():
        label_description[f"{key}_00_0"] = f"{crop[key]}_정상"
        for disease_code in value:
            for risk_code in risk:
                label = f"{key}_{disease_code}_{risk_code}"
                label_crop = f"{key}"
                label_disease = f"{disease_code}"
                label_risk = f"{risk_code}"

                label_description[
                    label
                ] = f"{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}"
                label_description_crop[label_crop] = f"{crop[key]}"
                label_description_disease[
                    label_disease
                ] = f"{disease[key][disease_code]}"
                label_description_risk[label_risk] = f"{risk[risk_code]}"

    label_description_disease["00"] = "정상"
    label_description_risk["0"] = "정상"

    # ex) '1_00_0' : 0
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_encoder_crop = {key: idx for idx, key in enumerate(label_description_crop)}
    label_encoder_disease = {
        key: idx for idx, key in enumerate(label_description_disease)
    }
    label_encoder_risk = {key: idx for idx, key in enumerate(label_description_risk)}

    # ex) '0' : '1_00_0'
    label_decoder = {val: key for key, val in label_encoder.items()}
    label_decoder_crop = {val: key for key, val in label_encoder_crop.items()}
    label_decoder_disease = {val: key for key, val in label_encoder_disease.items()}
    label_decoder_risk = {val: key for key, val in label_encoder_risk.items()}

    # print(label_decoder)
    if enc_dec_mode == 0:
        return csv_feature_dict, label_encoder, label_decoder
    else:
        return (
            csv_feature_dict,
            [label_encoder_crop, label_encoder_disease, label_encoder_risk],
            [label_decoder_crop, label_decoder_disease, label_decoder_risk],
        )
def predict(dataset):
    device = torch.device("cuda")
    # model = EfficientNet.from_name(model_name,num_classes=111)
    model = torch.load('/works/EffcientNetV2/weights/best_pt.pt', map_location='cuda')['model'].float().eval()
    # model = models.__dict__[model_name]()
    # model = EfficientNet.from_pretrained(model_name,"/works/EfficientNet-PyTorch/examples/imagenet/model_best_eff.pth")
    model.eval()
    model.to(device)
    tqdm_dataset = tqdm(enumerate(dataset))
    training = False
    results = []
    answer = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item["img"].to(device)
        # seq = batch_item["csv_feature"].to(args.gpu)
        with torch.no_grad():
            output = model(img)
        output = torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
        results.extend(output)
    return results
def main():
    _, label_encoder, label_decoder = csv_reader(enc_dec_mode=0)
    test = sorted(glob("/DL_data_big/DACON_2022_LeafChallenge/leaf_data/data/test/*"))
    test_dataset = CustomDataset(test,label_encoder,mode="test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=16, shuffle=False)    
    preds = predict(test_dataloader)
    preds = np.array([label_decoder[int(val)] for val in preds])
    submission = pd.read_csv("/works/EfficientNet-PyTorch/examples/imagenet/data/sample_submission.csv")
    submission["label"] = preds
    submission.to_csv("/works/EffcientNetV2/baseline_submission_efficientnetv2_rw_m_90.csv", index=False)

if __name__ == '__main__':
    main()