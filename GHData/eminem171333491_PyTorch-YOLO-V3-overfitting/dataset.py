import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
import cfg
import os
from PIL import Image
import math


IMG_BASE_DIR = "data"

transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
])

def one_hot(cls_num, i):
    b = np.zeros(cls_num)
    b[i] = 1.
    return b

class MyDataset(Dataset):

    def __init__(self,LABEL_FILE_PATH):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]
        strs = line.split()
        # print(os.path.join(IMG_BASE_DIR, strs[0]))
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))

        img_data = transforms(_img_data)
        # img_data = transforms(_img_data)
        # _boxes = np.array(float(x) for x in strs[1:])
        _boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)
                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])#4,i
        return labels[13], labels[26], labels[52], img_data
if __name__ == '__main__':
    x=one_hot(4,2)
    print(x)
    LABEL_FILE_PATH = r"data/the_label.txt"
    data = MyDataset(LABEL_FILE_PATH)
    dataloader = DataLoader(data,2,shuffle=True)
    for i,x in enumerate(dataloader):
        print(x[0].shape)
        print(x[1].shape)
        print(x[2].shape)
        print(x[3].shape)
    for target_13, target_26, target_52, img_data in dataloader:
        print(target_13.shape)
        print(target_26.shape)
        print(target_52.shape)
        print(img_data.shape)
