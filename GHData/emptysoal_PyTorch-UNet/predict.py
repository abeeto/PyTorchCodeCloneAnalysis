import torch
import torchvision
import cv2
import numpy as np

from torchvision.utils import save_image

from dataset import Datasets
from unet import UNet

if __name__ == '__main__':
    net = UNet()
    net.load_state_dict(torch.load(r'./model.plt'))

    trans = torchvision.transforms.ToTensor()

    img = cv2.imread('./self_data/train_images/JPEGImages/012.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Datasets.preprocess(img, 256)
    img = trans(img).numpy()[np.newaxis, :]
    img = torch.from_numpy(img)

    out = net(img)
    print(out.shape)

    save_image(out, r'./test_result/result.jpg', nrow=1)
