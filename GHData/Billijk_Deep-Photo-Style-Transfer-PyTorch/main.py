from __future__ import print_function
import argparse
import matlab
from matlab.engine import start_matlab
from PIL import Image
import numpy as np
import os

import torch
import torchvision.transforms.functional as t_func
import torchvision.models as models
import torch.optim as optim

from model import get_style_model_and_losses
from segment import add_arguments


parser = argparse.ArgumentParser()
parser.add_argument("content", type=str, help="Path of content image.")
parser.add_argument("style", type=str, help="Path of style image.")
parser.add_argument("output", type=str, help="Path of output image.")
parser.add_argument("--masks", type=str, help="Path of masks to load.")
parser.add_argument("--lr", type=float, default=1.0, help="Initial learning rate.")
parser.add_argument("--iters", type=int, default=300, help="Number of iterations to run.")
parser.add_argument("--size", type=int, default=480, help="Size for scaling image.")
parser.add_argument("--post_s", type=float, default=60.0, help="sigma_s for post processing recursive filter. (default: 60)")
parser.add_argument("--post_r", type=float, default=1.0, help="sigma_r for post processing recursive filter. (default: 1)")
parser.add_argument("--post_it", type=int, default=3, help="Number of iterations for post processing recursive filter. (default: 3)")
parser.add_argument("--ws", type=float, default=1e6, help="Weight for style loss (default: 10^6).")
parser.add_argument("--wc", type=float, default=1, help="Weight for content loss (default: 1).")
parser.add_argument("--wsim", type=float, default=10, help="Weight for similarity loss (default: 10).")
add_arguments(parser)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_name, h, w=None):
    image = Image.open(image_name)
    if w is not None: size = (h, w)
    else: size = h
    image = t_func.resize(image, size)
    # fake batch dimension required to fit network's input dimensions
    image = t_func.to_tensor(image).unsqueeze(0)
    return image.to(device, torch.float)


def load_masks(h, w):
    masks = None
    if args.masks is not None:
        # load masks
        masks = torch.load(args.masks)
        if style_mask.shape[1] != h or style_mask.shape[2] != w:
            print("Style mask shape is not compatible with desired image size ({}, {})".format(h, w))
            masks = None
        if content_mask.shape[1] != h or content_mask.shape[2] != w:
            print("Content mask shape is not compatible with desired image size ({}, {})".format(h, w))
            masks = None
    
    if masks is None:
        # create masks
        from segment import segment
        masks = segment(args, h, w)
    
    style_mask = masks["tar"]
    content_mask = masks["in"]
    style_mask = style_mask.to(device).unsqueeze(1)
    content_mask = content_mask.to(device).unsqueeze(1)
    return style_mask, content_mask


if __name__ == "__main__":

    style_img = image_loader(args.style, args.size)
    content_img = image_loader(args.content, style_img.size(2), style_img.size(3))

    print(style_img.size())
    print(content_img.size())

    assert style_img.size() == content_img.size(),     "we need to import style and content images of the same size"

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    style_mask, content_mask = load_masks(style_img.size(2), style_img.size(3))

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('Building the style transfer model..')
    model, style_losses, content_losses, sim_losses = get_style_model_and_losses(cnn,
        cnn_normalization_mean, cnn_normalization_std, style_img, content_img,
        style_mask, content_mask, device)
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=args.lr)

    print('Optimizing..')

    run = [0]
    while run[0] <= args.iters:

        def closure():

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            sim_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            for siml in sim_losses:
                sim_score += siml.loss

            style_score *= args.ws
            content_score *= args.wc
            sim_score *= args.wsim

            loss = style_score + content_score + sim_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print('run {}: Style Loss : {:4f} Content Loss: {:4f} Similarity Loss: {:4f}'.format(
                    run, style_score.item(), content_score.item(), sim_score.item()))

            return loss

        optimizer.step(closure)
        # correct the values of updated input image
        input_img.data.clamp_(0, 1)

    def unload(tensor):
        # Convert an image from tensor to PIL image
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image_pil = t_func.to_pil_image(image)
        return image_pil

    output = unload(input_img)
    save_path = args.output
    plt.imsave(save_path, output)
    print("Save image to {}".format(save_path))


    print("Post processing")
    inimg = np.array(unload(content_img))
    inimg_mat = matlab.int32(inimg.tolist())
    outimg = np.array(output)
    outimg_mat = matlab.int32(outimg.tolist())

    eng = start_matlab()
    processed_img = inimg - np.asarray(eng.RF(inimg_mat, args.post_s, args.post_r, args.post_it, inimg_mat)) + \
            np.asarray(eng.RF(outimg_mat, args.post_s, args.post_r, args.post_it, inimg_mat))
    processed_img = np.uint8(np.clip(processed_img, 0, 255))

    output_rt, output_ext = os.path.splitext(args.output)
    save_path = output_rt + "_post" + output_ext
    print("Save post processed image to {}".format(save_path))
    plt.imsave(save_path, Image.fromarray(processed_img))
    
