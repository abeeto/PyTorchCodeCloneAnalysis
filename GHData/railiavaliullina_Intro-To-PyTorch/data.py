import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST
from skimage.util import random_noise
from collections import Counter
from PIL import Image
from config import cfg


"""
УЛУЧШЕНИЯ:

1) вынести в отдельную папку
2) добавить документацию к каждой функции, следить за названиями функций
3) TODO: ...
"""


# аугментация-добавление шума через skimage(не нашла в transforms)
class AddNoise(object):
    def __init__(self, mode='s&p'):
        self.mode = mode

    def __call__(self, img):
        img = random_noise(np.asarray(img, dtype=np.float32), mode=self.mode)
        return img


class Resize(object):
    def __init__(self, size):
        if isinstance(size, tuple):
            self.new_h, self.new_w = size
        elif isinstance(size, int):
            self.new_h, self.new_w = size, size
        else:
            raise Exception('size type must be tuple or int')
        self.size = size

    def __call__(self, img):
        img = img.resize((self.new_w, self.new_h), Image.BILINEAR)
        return img


def get_data():
    transforms_train = transforms.Compose([
        Resize((32, 32)),
        transforms.RandomCrop(cfg.sz_crop),
        # TODO: добавить опционально использование аугментации
        #  (вынести в конфиг использовать или нет; вынести список аугментаций для использования)
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # смещение
        # AddNoise(),  # шум salt&pepper
        transforms.ToTensor()])
    transforms_test = transforms.Compose([
        transforms.ToTensor()])

    # TODO: если используется несколько датасетов, вынести в конфиг и добавить опциональное использование
    ds_train = KMNIST(root='./data', train=True, transform=transforms_train, download=False)
    ds_test = KMNIST(root='./data', train=False, transform=transforms_test, download=False)

    # TODO: вынести в отдельную функцию подсчет и вывод статистики
    # dataset statistics
    nb_classes = len(ds_train.classes)
    ds_train.nb_images = len(ds_train.train_labels)
    ds_train.nb_images_per_class = Counter(np.asarray(ds_train.train_labels))
    ds_test.nb_images = len(ds_test.test_labels)
    ds_test.nb_images_per_class = Counter(np.asarray(ds_test.test_labels))

    dl_train_ = DataLoader(dataset=ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_test_ = DataLoader(dataset=ds_test, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    print(f'Dataset statistics: ')
    print(f'number of classes: {nb_classes}')
    print(f'number of images in train set: {ds_train.nb_images}')
    print(f'number of images in test set: {ds_test.nb_images}')
    print(f'number of train images per class: {sorted(ds_train.nb_images_per_class.items())}')
    print(f'number of test images per class: {sorted(ds_test.nb_images_per_class.items())}')
    return dl_train_, dl_test_
