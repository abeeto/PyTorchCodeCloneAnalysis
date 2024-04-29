from torchvision.transforms import functional as F
from PIL import Image
import torch


class Transform(object):
    def __init__(self, height, width, img_mean, img_std):
        self.mean = img_mean
        self.std = img_std
        self.height = height
        self.width = width
        self.ratio = width / height

    def __call__(self, img):
        w, h = img.size
        ratio = w / h
        scale = self.width / w if ratio > self.ratio else self.height / h
        img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
        w, h = img.size
        img = F.to_tensor(img)
        resized_img = torch.zeros((3, self.height, self.width))
        x = int((self.width-w)/2)
        y = int((self.height-h)/2)
        resized_img[:, y:y+h, x:x+w] = img

        return F.normalize(resized_img, self.mean, self.std)
        # return resized_img


def calculate_mean_and_std(dataset_loader, dataset_size):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data in dataset_loader:
        now_batch_size, c, h, w = data[0].shape
        mean += torch.sum(torch.mean(torch.mean(data[0], dim=3), dim=2), dim=0)
        std += torch.sum(torch.std(data[0].view(now_batch_size, c, h * w), dim=2), dim=0)
    return mean/dataset_size, std/dataset_size