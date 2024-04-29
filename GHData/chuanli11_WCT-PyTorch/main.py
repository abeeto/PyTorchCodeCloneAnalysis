import argparse
import time
import os

from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from wct import WCT


parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--content',default='images/content/in1.jpg',help='path to content image')
parser.add_argument('--style',default='images/style/in1.jpg',help='path to style image')
parser.add_argument('--output', default='output/1.jpg', help='path to output image')
parser.add_argument('--alpha', type=float,default=1.0, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
parser.add_argument('--content_w', type=int, default=512, help="resized content image width")
parser.add_argument('--content_h', type=int, default=512, help="resized content image height")
parser.add_argument('--style_w', type=int, default=512, help="resized style image width")
parser.add_argument('--style_h', type=int, default=512, help="resized style image height")
args = parser.parse_args()


def styleTransfer(wct, alpha, contentImg, styleImg, csF, output):

    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = wct.transform(cF5,sF5,csF,alpha)
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(styleImg)
    cF4 = wct.e4(Im5)
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = wct.transform(cF4,sF4,csF,alpha)
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(styleImg)
    cF3 = wct.e3(Im4)
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transform(cF3,sF3,csF,alpha)
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(styleImg)
    cF2 = wct.e2(Im3)
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transform(cF2,sF2,csF,alpha)
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(Im2)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform(cF1,sF1,csF,alpha)
    Im1 = wct.d1(csF1)

    vutils.save_image(Im1.data.cpu().float(), output)
    return


def main():
    # Prepare WCT model
    vgg1 = 'models/vgg_normalised_conv1_1.pth'
    vgg2 = 'models/vgg_normalised_conv2_1.pth'
    vgg3 = 'models/vgg_normalised_conv3_1.pth'
    vgg4 = 'models/vgg_normalised_conv4_1.pth'
    vgg5 = 'models/vgg_normalised_conv5_1.pth'
    decoder1 = 'models/feature_invertor_conv1_1.pth'
    decoder2 = 'models/feature_invertor_conv2_1.pth'
    decoder3 = 'models/feature_invertor_conv3_1.pth'
    decoder4 = 'models/feature_invertor_conv4_1.pth'
    decoder5 = 'models/feature_invertor_conv5_1.pth'
    paths = vgg1, vgg2, vgg3, vgg4, vgg5, decoder1, decoder2, decoder3, decoder4, decoder5
    wct = WCT(paths)

    # Prepare images
    content_image = Image.open(args.content).resize((args.content_w, args.content_h))
    contentImg = TF.to_tensor(content_image)
    contentImg.unsqueeze_(0)
    style_image = Image.open(args.style).resize((args.style_w, args.style_h))
    styleImg = TF.to_tensor(style_image)
    styleImg.unsqueeze_(0)
    csF = torch.Tensor()
    
    cImg = Variable(contentImg, volatile=True)
    sImg = Variable(styleImg, volatile=True)
    csF = Variable(csF)

    cImg = cImg.cuda(0)
    sImg = sImg.cuda(0)
    csF = csF.cuda(0)
    wct.cuda(0)

    
    # Run style transfer
    start_time = time.time()
    styleTransfer(wct, args.alpha, cImg, sImg, csF, args.output)
    end_time = time.time()
    print('Elapsed time is: %f' % (end_time - start_time))    


if __name__ == "__main__":
    main()