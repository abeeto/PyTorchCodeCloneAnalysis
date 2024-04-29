from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image


class ConcatDatasets(torch.utils.data.Dataset):
    def __init__(self, image_folders, transform=None):
        self.folders = [Path(folder) for folder in image_folders]
        self.classes = [child.name for child in sorted(self.folders[0].iterdir())]
        self.data = []
        self.transform = transform
        for folder in tqdm(self.folders, desc='Reading folders...'):
            for i, child in enumerate(sorted(folder.iterdir())):
                assert child.name in self.classes, 'Currently, datasets with different classes are not supported!'
                for name in child.iterdir():
                    if name.is_file():
                        self.data.append((str(name), i))
        self.indices = list(range(len(self.data)))
        print(f'Successfully loaded {len(image_folders)} datasets with total {len(self.data)} images.\nClasses: {self.classes}')

    def __len__(self):
        return len(self.data)

    def get_calibration_data(self):
        class_labels = []
        for idx in tqdm(self.indices, desc='Getting label statistics...'):
            class_labels.append(self.get_label_by_index(idx))
        class_weights = 1 / torch.Tensor(class_labels).unique(return_counts=True)[1]
        class_weights /= class_weights.sum()
        return class_labels, class_weights

    def get_label_by_index(self, idx):
        return self.data[idx][1]

    def get_indexes_by_class_id(self, cls_id):
        ret = []
        for i, entry in enumerate(self.data):
            if (entry[1] == cls_id):
                ret.append(i)
        return ret

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        with open(img_name, 'rb') as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label
