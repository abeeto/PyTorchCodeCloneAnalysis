import numpy as np
import PIL
import torch


class ImagesDataset:
    def __init__(self,
        paths_of_images) -> None:
        self.paths_of_images = paths_of_images

    def __len__(self) -> int:
        return len(self.paths_of_images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.paths_of_images[idx]
        image = PIL.Image.open(image_path)
        image = image.convert("RGB") # if we have a single channel image for example
        return {
            "image": np.array(image)
        }


class TorchDataset:
    def __init__(self, images_dataset: ImagesDataset) -> None:
        self.images_dataset = images_dataset

    def __len__(self) -> int:
        return len(self.images_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.images_dataset[idx]
        image = np.transpose(item['image'], (2, 0, 1)).astype(np.float32) # pytorch expects CHW
        item['image'] = torch.tensor(image)
        return item


class ClassificationDataset:
    def __init__(self, images_dataset: TorchDataset, targets) -> None:
        assert len(images_dataset) == len(targets)
        self.images_dataset = images_dataset
        self.targets = targets

    def __len__(self) -> int:
        return len(self.images_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.images_dataset[idx]
        targets = self.targets[idx]
        item.update({
            'targets': torch.tensor(targets, dtype=torch.long)
        })
        return item


class AugmentedDataset(ImagesDataset):
    def __init__(self, images_dataset: ImagesDataset, augmentations) -> None:
        self.images_dataset = images_dataset
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.images_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.images_dataset[idx]
        augmented = self.augmentations(image=item['image'])
        item['image'] = augmented['image']
        return item

"""
Example usage:

def create_dataset(paths_of_images, targets, augmentations):
    ds = ImagesDataset(paths_of_images)
    ds = AugmentedDataset(ds, augmentations)
    ds = TorchDataset(ds)
    return ClassificationDataset(ds, targets)
"""