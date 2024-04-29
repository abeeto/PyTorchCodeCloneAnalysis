# coding : utf-8

import os
import sys
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import create_model
from tqdm import tqdm


class Demo(object):
    def __init__(self, model_path, device='cuda'):
        self._device = device

        self._model = create_model(model_name='mixnet_s', num_classes=4, pretrained=False, in_chans=3, global_pool='avg')
        for param in self._model.parameters():
            param.requires_grad = False

        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        for k, v in list(state_dict.items()):
            state_dict[k.replace('module.', '')] = state_dict.pop(k)

        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        self._angle = [0, 90, 180, 270]

    def _center_crop(self, image, crop_size, crop_scale):
        scale_size = tuple([int(x / crop_scale) for x in crop_size])
        img = cv2.resize(image, dsize=scale_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[0:2]
        sx = int(round((w - crop_size[0]) / 2.))
        sy = int(round((h - crop_size[1]) / 2.))
        img = img[sy:sy+crop_size[1], sx:sx+crop_size[0], :]
        return img

    def predict(self, image):
        img = self._center_crop(image, crop_size=(224, 224), crop_scale=0.875)
        img = self._transform(img).to(self._device)
        with torch.set_grad_enabled(False):
            pred = self._model(img[None, :, :, :])
            pred = F.softmax(pred, dim=1)
        score, index = pred.max(dim=1)
        score = score.data.cpu().numpy()[0]
        index = index.data.cpu().numpy()[0]
        angle = self._angle[index]
        if angle > 0:
            image = np.rot90(image, k=int(angle)//90)
        return angle, image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--images_dir', type=str, required=True, help='The path of images for testing')
    parser.add_argument('--output_dir', type=str, required=True, help='The path to save predicting images')
    args = parser.parse_args()

    demo = Demo(args.model_path, device='cuda')
    file_list = os.listdir(args.images_dir)
    for index, file in enumerate(tqdm(file_list, ncols=100)):
        file_name = os.path.join(args.images_dir, file)
        try:
            img_buf = np.fromfile(file_name, dtype=np.uint8)
            image = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            if image is None:
                continue
            angle, _ = demo.predict(image)
            saved_dir = os.path.join(args.output_dir, str(angle))
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)
            cv2.imwrite(os.path.join(saved_dir, file), image)
        except Exception as e:
            print('{}: {}'.format(file, e))
            continue

