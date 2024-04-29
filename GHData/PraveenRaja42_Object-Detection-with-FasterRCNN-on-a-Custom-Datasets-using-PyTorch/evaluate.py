from fastcore.test import test
from matplotlib.pyplot import text
import torch
from torch.utils import data
from torch_snippets import *
import data_preprocessing
import model
import os
import argparse
import torch.optim as optim
from torchvision.ops import nms
from PIL import Image


DEFAULT_IMG_DIR = "C:/Users/91950/Object_detection_project/images"
DEFAULT_DF_DIR = "C:/Users/91950/Object_detection_project/df.csv"


def decode_output(output, df):
    df = pd.read_csv(df)
    label2target = {l: t + 1 for t, l in enumerate(df["LabelName"].unique())}
    label2target["background"] = 0
    target2label = {t: l for l, t in label2target.items()}
    num_classes = len(label2target)
    "convert tensors to numpy arrays"
    bbs = output["boxes"].cpu().detach().numpy().astype(np.uint16)
    labels = np.array(
        [target2label[i] for i in output["labels"].cpu().detach().numpy()]
    )
    confs = output["scores"].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serialized_file", required=True, help="saved model weights")
    parser.add_argument(
        "--img_dir", default=DEFAULT_IMG_DIR, help="dir to image folder"
    )
    parser.add_argument(
        "--df_dir", default=DEFAULT_DF_DIR, help="dir to metadata folder"
    )
    parser.add_argument(
        "--save_dir", required=True, help="dir to save images"
    )
    args = parser.parse_args()

    _, test_loader = data_preprocessing.data_loader(args.df_dir, args.img_dir)
    model = model.get_model()
    model.load_state_dict(torch.load(args.serialized_file))
    model.eval()

    for ix, (images, targets) in enumerate(test_loader):
        if ix == 3:
            break
        images = [im for im in images]
        output = model(images)
        for ix, output in enumerate(output):
            bbs, confs, labels = decode_output(output, args.df_dir)
            info = [f"{l}@{c:.2f}" for l, c in zip(labels, confs)]
            show(
                images[ix].cpu().permute(1, 2, 0),
                bbs=bbs,
                texts=labels,
                sz=5,
                save_path=args.save_dir/f"Image{ix}",
            )
