import os
from PIL import Image
from io import BytesIO

from typing import List, Tuple

import numpy as np

import torch
from torchvision import transforms


def enumerate_images(images_dir: str) -> List[str]:
    images_list: List[str] = []

    for dirs, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".ifjf")):
                images_list.append(os.path.join(dirs, file))

    return images_list


def compare_faces(features_1: np.ndarray, feature_2: np.ndarray, threshold: float=0.8) -> bool:
    assert len(feature_2.shape) == 1
    assert len(features_1.shape) == 2
    assert features_1.shape[1] == feature_2.shape[0]

    '''similarity = np.sum(features_1 * feature_2[np.newaxis, :], axis=1)

    return (similarity > threshold)'''
    similarity = np.sqrt(np.sum((features_1 - feature_2[np.newaxis, :])**2, axis=1))
    return (similarity < threshold)


def load_images_and_labels_into_tensors(images_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    images_list = enumerate_images(images_dir=images_dir)

    transform = transforms.Compose([transforms.Resize([112, 112]), transforms.ToTensor()])

    class_list = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], images_list))))
    class_list.sort()

    class_to_label = dict(zip(class_list, range(len(class_list))))

    images = []
    labels = []

    for image in images_list:
        label = int(class_to_label[os.path.normpath(image).split(os.sep)[-2]])
        labels.append(label)

        img = Image.open(image).convert("RGB")

        img = transform(img)

        images.append(img)

    images = torch.stack(images, dim=0)
    labels = torch.from_numpy(np.asarray(labels))

    return images, labels


def read_image(image_encoded: bytes, mode="RGB") -> np.ndarray:
    pil_image = Image.open(BytesIO(image_encoded))
    pil_image.convert(mode)
    img = np.array(pil_image)

    return img


'''images, labels = load_images_and_labels_into_tensors(images_dir=r"D:\Face_Datasets\choose_train")
print(images.min(), images.max(), images.shape, labels.shape)'''
