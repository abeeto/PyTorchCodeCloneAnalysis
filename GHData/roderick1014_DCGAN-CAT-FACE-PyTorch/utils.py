import torch
import os
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image

def save_checkpoint(model , optimizer , filename = "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint , filename)


def load_checkpoint(checkpoint_file , model , optimizer , lr):
    print("=> Loading checkpoint")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_file , map_location = DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(FILE , test_folder , img):
    save_image(img  , f"TestFolder/{FILE}")
    
def plot_reals(FILE , test_folder , img):
    save_image(img  , f"TestFolder/{FILE}")
