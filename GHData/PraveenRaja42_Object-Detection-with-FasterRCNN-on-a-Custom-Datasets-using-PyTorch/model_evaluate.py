from fastcore.test import test
from matplotlib.pyplot import text
import torch
import glob
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
DEFAULT_SAVE_DIR = "saves"


def preprocess_image(img):
    img = torch.tensor(img).permute(2, 0, 1)
    return img.float()


def data_preprocess(img_path):
    img_files = glob.glob(img_path + "/*")
    res_img = []
    for img in img_files:

        ig = Image.open(img).convert("RGB")
        ig = np.array(ig.resize((224, 224), resample=Image.BILINEAR)) / 255
        ig = preprocess_image(ig)
        res_img.append(ig)
    return res_img


def decode_output(output):
    target2label = {1: "Bus", 2: "Truck", 0: "background"}
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
        "--save_path", default=DEFAULT_SAVE_DIR, help="dir to save result images"
    )
    args = parser.parse_args()

    img_files = data_preprocess(args.img_dir)
    test_loader = DataLoader(img_files, batch_size=len(img_files), drop_last=True)
    model = model.get_model()
    model.load_state_dict(torch.load(args.serialized_file))
    device = torch.device("cpu")
    model.eval()

    for ix, images in enumerate(test_loader):
        # if ix==3: break
        # images = [im for im in images]
        outputs = model(images)
        for ix, output in enumerate(outputs):
            bbs, confs, labels = decode_output(output)
            info = [f"{l}@{c:.2f}" for l, c in zip(labels, confs)]
            show(
                images[ix].cpu().permute(1, 2, 0),
                bbs=bbs,
                texts=labels,
                sz=5,
                save_path=os.path.join(args.save_path, f"res_img {ix}"),
            )
