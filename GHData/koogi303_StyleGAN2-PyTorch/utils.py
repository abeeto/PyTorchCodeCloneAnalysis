import torch
import numpy as np
from pytorch_msssim import ssim
import PIL.Image as pil_image
import random

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def check_image_file(filename: str):
    return any(
        filename.endswith(extension)
        for extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
            ".JPG",
            ".JPEG",
            ".PNG",
        ]
    )


# 전처리 과정 함수
def preprocess(img):
    # uInt8 -> float32로 변환
    x = np.array(img).astype(np.float32)
    x = x.transpose([2, 0, 1])
    # Normalize x 값
    x /= 255.0
    # 넘파이 x를 텐서로 변환
    x = torch.from_numpy(x)
    # x의 차원의 수 증가
    x = x.unsqueeze(0)
    # x 값 반환
    return x


def calc_psnr(img1, img2):
    """PSNR 계산"""
    return 10.0 * torch.log10(1.0 / torch.mean((img1 - img2) ** 2))


def calc_ssim(img1, img2):
    """SSIM 계산"""
    return ssim(img1, img2, data_range=1, size_average=False)



def get_concat_h(img1, img2):
    merge = pil_image.new("RGB", (img1.width + img2.width, img1.height))
    merge.paste(img1, (0, 0))
    merge.paste(img2, (img2.width, 0))
    return merge


def get_concat_v(img1, img2):
    merge = pil_image.new("RGB", (img1.width, img1.height + img2.height))
    merge.paste(img1, (0, 0))
    merge.paste(img2, (0, img2.height))
    return merge


# Copy from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

