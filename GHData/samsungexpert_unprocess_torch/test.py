import argparse
import os
import torch
import numpy as np
import cv2

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset
from models import DemosaicNet
from utils import init_weight, ImagePool, LossDisplayer, rgb_augment

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset_path", type=str, default="D:\\MIT_dataset\\test\\")
parser.add_argument("--checkpoint_path", type=str, default=".\\checkpoint\\150.pth")

args = parser.parse_args()

def save_img(img, filename):
    img = np.squeeze(img, axis=0)
    img = img.transpose(1,2,0)
    img = (255 * img)
    img = img.astype(np.uint8)
    cv2.imwrite(filename, img)
    


def test():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    print(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataloader = DataLoader(
        Dataset(args.dataset_path,transform)
    )

    model = DemosaicNet()
    model.load_state_dict(torch.load(args.checkpoint_path)['net_state_dict'])

    with torch.no_grad():

        cnt = 0
        for data in dataloader:
            image_in, ref_image = rgb_augment(data)
            # calculate outputs by running images through the network
            outputs = model(image_in)

            aa = outputs.detach().numpy()
            bb = ref_image.detach().numpy()

            print(aa.shape, bb.shape)

            
            save_img(aa, "output.png")
            save_img(bb, "ref.png")


            cnt +=1 
            if cnt == 1:
                break

if __name__ == "__main__":
    test()
