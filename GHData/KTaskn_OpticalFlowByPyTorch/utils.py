import torch
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np

def open_image(path):
    return TF.to_tensor(Image.open(path))


def open_image_grayscale(path):
    return TF.to_tensor(Image.open(path).convert('L'))

def preview(path: str, flows: torch.tensor, step=5):
    h, w = flows.size(0), flows.size(1)
    y, x = torch.meshgrid(torch.arange(0, h, 5), torch.arange(0, w, 5), indexing='ij')
    y, x = y.ravel(), x.ravel()
    u, v = flows.permute(2, 0, 1)[:, ::step, ::step].reshape(2, -1)
    lines = torch.vstack([x, y, x + v, y + u]).T.view(-1, 2, 2).int().numpy()

    return Image.fromarray(
        cv2.polylines(
            np.array(Image.open(path)),
            lines,
            0,
            (0, 255, 0)
        ))