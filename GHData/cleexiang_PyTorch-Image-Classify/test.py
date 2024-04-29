from efficientnet import EfficientNet
import torch
import torch.nn as nn
from PIL import Image
import argparse
from dataset import build_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./data/val/others/9.jpg', dest='path')
    args = parser.parse_args()
    model = EfficientNet.from_name('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('./checkpoints/checkpoint.pth.tar.epoch_19'))
    model.eval()
    image = Image.open(args.path)
    transform = build_transform(224)
    input_tensor = transform(image).unsqueeze(0)
    pred = model(input_tensor)
    # pred = model(input_tensor).argmax()
    print("prediction:", pred)
