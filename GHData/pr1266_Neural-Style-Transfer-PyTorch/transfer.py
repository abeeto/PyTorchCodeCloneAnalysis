from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from utils import load_image
os.system('cls')

#! config target device if we use cuda or cpu:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGGNet(nn.Module):
    def __init__(self):
        #! select certain activation channels:
        super(VGGNet, self).__init__()
        #! here we select 0, 5, 10, 19 and 28th layer of vgg network
        self.select = ['0', '5', '10', '19', '28'] 
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x):
        #! extract and store output of considered layers:
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def main(config):
   
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )
    
    
    content = load_image(config.content, transform, max_size=config.max_size)
    style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])
    
    #! target is initialized with original content image at first:
    #! we can also use a random image (noisy)
    # target = content.clone().requires_grad_(True)
    target = torch.randn(content.shape, device=device, requires_grad=True)
    #! we pass "target image" parameters to our optimizer:
    optimizer = torch.optim.Adam([target], lr=config.lr, betas=[0.5, 0.999])
    #! we set model state to eval because we dont need to change it weights
    #! instead, we are going to apply backward propagation on target image
    #! and change it to a mixture of content and style images based on alpha and beta
    vgg = VGGNet().to(device).eval()
    
    for step in range(config.total_step):
        
        #! first we feed our source, target and style image to our network
        #! and then we store the certain layers output as our features:
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)
        style_loss = 0
        content_loss = 0

        #! now we iterate over extracted features:
        for target_feature, content_feature, style_feature in zip(target_features, content_features, style_features):
            #! Compute content loss with target and content images
            content_loss += torch.mean((target_feature - content_feature)**2)

            b, c, h, w = target_feature.size()
            #! here we reshape style and target features to a c x (h*w)
            #! actually we create c vectors in size (h*w) to calculate gram matrix
            #! gram matrix calculates correlation of elements of a matrix:
            target_feature = target_feature.view(c, h*w)
            style_feature = style_feature.view(c, h*w)

            #! gram matrix calculation:
            target_feature = torch.mm(target_feature, target_feature.t())
            style_feature = torch.mm(style_feature, style_feature.t())

            #! Compute style loss with target and style images
            style_loss += torch.mean((target_feature - style_feature)**2) / (c*h*w) 
        
        #! here we calculate overall loss based on loss function mentioned in paper
        loss = config.content_weight * content_loss + config.style_weight * style_loss 
        #! back propagation stage:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % config.log_step == 0:
            print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}' 
                   .format(step+1, config.total_step, content_loss.item(), style_loss.item()))

        if (step+1) % config.sample_step == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-{}.png'.format(step+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='png/src.jfif')
    parser.add_argument('--style', type=str, default='png/style2.jpg')
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--total_step', type=int, default=10000)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--style_weight', type=float, default=70)
    parser.add_argument('--content_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.02)
    config = parser.parse_args()
    print(config)
    main(config)