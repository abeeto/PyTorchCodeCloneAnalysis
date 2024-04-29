import pandas as pd
import numpy as np
from scipy.stats import rankdata


def ranks(array):
    return array
    return rankdata(array, method="min") / len(array)
    # temp = np.argsort(array) / len(array)
    # ranks = np.empty_like(temp)
    # ranks[temp] = np.arange(len(array)) / len(array)
    # return ranks

if __name__ == "__main__":
    p1 = pd.read_csv("exp/melanoma7/predict0.csv")
    p2 = pd.read_csv("exp/melanoma7/predict1.csv")
    p3 = pd.read_csv("exp/melanoma7/predict2.csv")
    p4 = pd.read_csv("exp/melanoma7/predict3.csv")
    p5 = pd.read_csv("exp/melanoma7/predict4.csv")
    # p6 = pd.read_csv("predict5_5.csv")
    ens = ranks(p1["target"]) *0.2 + ranks(p2["target"]) *0.2 + ranks(p3["target"]) *0.2 +ranks(p4["target"]) *0.2 + ranks(p5["target"]) * 0.2
    pred = pd.DataFrame(columns=["image_name", "target"])
    pred["image_name"] = p1["image_name"]
    pred["target"] = ens
    pred.to_csv("exp/melanoma7/predict.csv", index=None)
    # p1 = pd.read_csv("exp/melanoma4/predict_blend.csv")
    # p2 = pd.read_csv("exp/melanoma4/predict.csv")
    # p3 = pd.read_csv("exp/melanoma5/predict.csv")
    # p4 = pd.read_csv("exp/melanoma6/predict.csv")
    # ens = ranks(p1["target"]) * 0.1 + ranks(p2["target"]) * 0.3 + ranks(p3["target"]) * 0.3 * p4["target"] * 0.3
    # pred = pd.DataFrame(columns=["image_name", "target"])
    # pred["image_name"] = p1["image_name"]
    # pred["target"] = ens
    # pred.to_csv("exp/melanoma5/predict_blend.csv", index=None)