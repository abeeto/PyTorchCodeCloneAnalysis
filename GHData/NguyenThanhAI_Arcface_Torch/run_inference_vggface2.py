import os
from re import L

from typing import List

import numpy as np

from PIL import Image

import torch

import vggface2_models.resnet as ResNet
import vggface2_models.senet as SENet

from vggface2_models.utils import load_state_dict

from utils_fn import enumerate_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = r"C:\Users\Thanh\Downloads\resnet50_ft_weight.pkl"
images_dir = r"D:\Face_Datasets\choose_train"

images_list: List[str] = enumerate_images(images_dir=images_dir)

model = ResNet.resnet50(num_classes=8631, include_top=False)

load_state_dict(model, weights)

model.eval()
model.to(device)

mean_bgr = np.array([91.4953, 103.8827, 131.0912])

img = Image.open(images_list[9]).convert("RGB").resize((224, 224))
img = np.array(img)

img = np.transpose(img, (2, 0, 1))
img = img.astype(np.float32)
img -= mean_bgr[:, np.newaxis, np.newaxis]
img = torch.from_numpy(img).unsqueeze(0).float()
#img.div_(255).sub_(0.5).div_(0.5)

img = img.to(device=device)

print(img.shape)

with torch.no_grad():
    feat_1 = model(img).cpu().numpy()[0]
    
print(feat_1.shape)