import argparse
import logging
import os
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
#from PIL import Image
import imageio
from torchvision import transforms

from unet import UNet
from utils.dataset import BasicDataset



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        probs = F.softmax(output, dim=1)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.shape[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(in_files):
    out_files = []
    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append("{}_OUT{}".format(pathsplit[0].replace("/images/or/","/detections/"), pathsplit[1])) 
    return out_files


def mask_to_image(mask):
    return np.argmax(mask * 255, axis=0).astype(np.uint8)
    
def toRGB(predicted_classes, width, height):
    predicted_rgb = np.zeros((width, height, 3))
    for ii in range(width):
        for jj in range(height):
            if predicted_classes[ii,jj] == 0:
                predicted_rgb[ii,jj,0] = 240
                predicted_rgb[ii,jj,1] = 228
                predicted_rgb[ii,jj,2] = 66

            elif predicted_classes[ii,jj] == 1:
                predicted_rgb[ii,jj,0] = 86
                predicted_rgb[ii,jj,1] = 180
                predicted_rgb[ii,jj,2] = 233

            elif predicted_classes[ii,jj] == 2:
                predicted_rgb[ii,jj,0] = 0
                predicted_rgb[ii,jj,1] = 158
                predicted_rgb[ii,jj,2] = 115

            elif predicted_classes[ii,jj] == 3:
                predicted_rgb[ii,jj,0] = 0
                predicted_rgb[ii,jj,1] = 0
                predicted_rgb[ii,jj,2] = 0

            else:
                predicted_rgb[ii,jj,0] = 0
                predicted_rgb[ii,jj,1] = 255
                predicted_rgb[ii,jj,2] = 0
    predicted_rgb = predicted_rgb.astype(np.uint8)
    return predicted_rgb

if __name__ == "__main__":
    args = get_args()
    in_files = glob.glob( './images/or/*.tif')
    out_files = get_output_filenames(in_files)

    net = UNet(n_channels=10, n_classes=4, bilinear=False)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    start = time.time()
    for i, fn in enumerate(in_files):
        
        print(fn)
        logging.info("\nPredicting image {} ...".format(fn))

        # Open image
        img = imageio.imread(fn)
        # Predict image
        startpred= time.time()
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1.0,
                           out_threshold=args.mask_threshold,
                           device=device)

        endpred=time.time()
        print("\n\t time:",str(endpred-startpred))

        # Save grayscale and rgb images
        out_fn = out_files[i].replace('.tif','.png')
        result = mask_to_image(mask)
        print(out_fn, result.shape)
        imageio.imwrite(out_fn,result)
        imageio.imwrite(out_fn,toRGB(result, 256,256))

        logging.info("Mask saved to {}".format(out_fn))


    end = time.time()
    print("Tiempo predicciones: ",str(end-start))
