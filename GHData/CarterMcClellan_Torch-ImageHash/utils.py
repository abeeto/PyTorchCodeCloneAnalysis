import torch
import numpy as np

def pil_2_tensor(pil_img):
    X = torch.as_tensor(np.array(pil_img, copy=True))
    X = X.view(pil_img.size[0], pil_img.size[1], len(pil_img.getbands())) 
    return X.permute((2, 0, 1))
