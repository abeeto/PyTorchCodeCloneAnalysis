import os
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np

from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip
)

from model import Efficientnet
from model import Srnet

data_dir = '/steganalysis/dataset'
device = 'cuda'
model = Efficientnet().to(device)
img_size = 512

AUGMENTATIONS_TEST = Compose([
    Resize(img_size, img_size, p=1),
    ToFloat(max_value=255),
    ToTensorV2()
], p=1)


class Alaska2TestDataset(Dataset):

    def __init__(self, df, augmentations=None):
        self.data = df
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn = self.data.loc[idx][0]
        im = cv2.imread(fn)[:, :, ::-1]

        if self.augment:
            im = self.augment(image=im)

        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({'ImageFileName': list(test_filenames)}, columns=['ImageFileName'])

batch_size = 16
num_workers = 4
test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False,
                                          drop_last=False)

model.load_state_dict(torch.load('./EfficientNet/epoch_20_val_loss_13.7_auc_0.812.pth'))
model.eval()

preds = []
tk0 = tqdm(test_loader, disable=True)
with torch.no_grad():
    for i, im in enumerate(tk0):
        inputs = im["image"].to(device)
        # flip vertical
        im = inputs.flip(2)
        outputs = model(im)
        # fliplr
        im = inputs.flip(3)
        outputs = (0.25 * outputs + 0.25 * model(im))
        outputs = (outputs + 0.5 * model(inputs))
        preds.extend(F.softmax(outputs, 1).cpu().numpy())

preds = np.array(preds)
labels = preds.argmax(1)
new_preds = np.zeros((len(preds),))
temp = preds[labels != 0, 1:]
# new_preds[labels != 0] = [temp[i, val] for i, val in enumerate(temp.argmax(1))]
new_preds[labels != 0] = temp.sum(1)
new_preds[labels == 0] = preds[labels == 0, 0]

test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
test_df['Label'] = new_preds

test_df = test_df.drop('ImageFileName', axis=1)
test_df.to_csv('submission.csv', index=False)
print(test_df.head())
