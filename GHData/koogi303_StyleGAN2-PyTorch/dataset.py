import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from utils import check_image_file

# 데이터 셋 생성 클래스
class Dataset(object):
    def __init__(self, images_dir, image_size):
        """이미지 파일 불러오기"""
        self.filenames = [
            os.path.join(images_dir, x)
            for x in os.listdir(images_dir)
            if check_image_file(x)
        ]

        self.augment_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

    def __getitem__(self, idx):
        hr = self.augment_transform(Image.open(self.filenames[idx]).convert("RGB"))
        return hr

    def __len__(self):
        return len(self.filenames)