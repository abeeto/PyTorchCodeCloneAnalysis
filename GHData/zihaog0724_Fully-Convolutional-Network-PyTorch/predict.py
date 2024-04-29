import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from fcn_dataset import SegDataset, img_transforms, BGR2RGB
from fcn_model import fcn_8x_resnet34
from utils import background_subtraction, noise_reduction
import cv2
import os

device = torch.device("cuda")


val_dataset = SegDataset(False, img_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

model = fcn_8x_resnet34()
model.to(device)
model.load_state_dict(torch.load("PATH_TO_PTH"))

root = "PATH_TO_IMGS"
img_list = os.listdir(root)
model.eval()
with torch.no_grad():
    for img_name in img_list:
        print(img_name)
        img = cv2.imread(os.path.join(root, img_name))
        img = cv2.resize(img, (500, 500), cv2.INTER_LINEAR)
        raw_img = cv2.resize(img, (512, 512), cv2.INTER_LINEAR).transpose(2, 0, 1)
        #raw_img = img.copy()
        img = BGR2RGB(img)
        img, label = img_transforms(img, None, inference=True)
        img = img.unsqueeze(0).float().to(device)
        output = model(img)
        output = torch.sigmoid(output)
        output = output.max(dim=1)[1].data.cpu().numpy()
        output = noise_reduction(output[0])
        res = background_subtraction(raw_img, output)
        cv2.imwrite("./test_res/" + img_name, res)