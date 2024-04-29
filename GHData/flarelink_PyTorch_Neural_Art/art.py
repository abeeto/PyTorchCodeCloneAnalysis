# =============================================================================
# art.py - Neural art in PyTorch
# Copyright (C) 2018  Humza Syed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

from __future__ import print_function, division

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchvision.models import vgg19, vgg16
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import argparse
from datetime import datetime
import random
import io
import requests
from PIL import Image
plt.ion() 


# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device, " the GPU device is selected")
else:
    device = torch.device('cpu')
    print(device, " the CPU device is selected")


"""
##############################################################################
# Image Functions
##############################################################################
"""
def load_imgs(cont_img_loc, cont_on, style_img_loc, style_on, out_img_size):
    """
    Load content and style images
    :param cont_img_loc:   content img to have neural art applied to;
                           will either be a URL or location to the img
    :param cont_on:        determine if image is from online or local
                           to determine how to load img
    :param style_img_loc:  style img to have neural art used with;
                           will either be a URL or location to the img
    :param style_on:       determine if image is from online or local
                           to determine how to load img
    :param out_img_size:   output image size
    :return: loading images
    """

    # transform data
    # resize img down to 256 for crop
    # center crop to 224 for VGG model
    # to tensor for GPU processing
    trans = transforms.Compose([
        transforms.Resize(out_img_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]
    )])

    # create pillow image from content image
    if(cont_on):
      cont_img = requests.get(cont_img_loc)
      pil_cont = Image.open(io.BytesIO(cont_img.content))
    else:
      pil_cont = Image.open(cont_img_loc)
    orig_cont = pil_cont

    # create pillow image from style image
    if(style_on):
      style_img = requests.get(style_img_loc)
      pil_style = Image.open(io.BytesIO(style_img.content))
    else:
      pil_style = Image.open(style_img_loc)
    orig_style = pil_style

    # apply transforms to imgs
    cont_trans = trans(pil_cont).unsqueeze(0).to(device)
    style_trans = trans(pil_style).unsqueeze(0).to(device)
    
    # create new image that will become stylized content image
    new_trans = cont_trans.clone().requires_grad_(True)

    return orig_cont, orig_style, cont_trans, style_trans, new_trans

    
def imshow(img, title='img', orig=False):
    """
    Function to show images
    param img:   input image to show
    param title: title of image
    param final:  boolean flag to check if this is the original content/style image
    """
    if(orig == False):
      transform = transforms.ToPILImage()
      plot_img = img.squeeze(0).cpu()
      plot_img = transform(plot_img)
    else:
      plot_img = img
    plt.imshow(plot_img)
    plt.title(title)
    plt.pause(1)
    plt.close()


"""
##############################################################################
# Feature Map Extraction
##############################################################################
"""

class Feature_Extraction(nn.Module):
  def __init__(self):
    # extract conv layer activation maps
    super(Feature_Extraction, self).__init__()
    self.sel_conv = ['0', '5', '10', '19', '28']
    self.vgg = vgg19(pretrained=True).features
  
  def forward(self, x):
    features = []
    for name, layer in self.vgg._modules.items():
      x = layer(x)
      if name in self.sel_conv:
        features.append(x)
    return features


"""
##############################################################################
# Loss Functions
##############################################################################
"""
# function for content image loss compared to new output image
def content_loss(F, P):
  """
  Obtains content loss 
  param: P = content image feature map in layer l
  param: F = new image feature map in layer l
  return content loss of images
  """
  loss = torch.mean((F - P)**2)
  return loss

def gram_matrix(F):
  """
  param F: feature map
  return gram matrix of feature map
  """
  gm = torch.mm(F, F.t())
  return gm

# function for style image loss compared to new output image
def style_loss(F, S, n, h, w):
  """
  param F: feature map of new img
  param S: feature map of style img
  param n: number of feature maps
  param h: height of feature maps
  param w: width of feature maps
  return style loss of images
  """
  loss = torch.mean((F - S)**2) / (n * h * w)
  return loss


"""
##############################################################################
# Training
##############################################################################
"""
def training(model, optimizer, n_epochs, content_img, style_img, new_img, style_weights, name):
  """
  Train from style and content images to make new image
  """
  for epoch in range(n_epochs):
    # get features from each img
    cont_feat = model(content_img)
    style_feat = model(style_img)
    new_feat = model(new_img)

    # initial loss set to 0
    cont_loss = 0
    sty_loss = 0

    # run through feature maps of each image
    for cf, sf, nf in zip(cont_feat, style_feat, new_feat):
      # compute content loss
      cont_loss += content_loss(nf, cf)

      # extract feature map shapes from new image
      b, n, h, w = nf.size() # b=batch size; c = number of feature maps; (h, w) = feature map dimensions, so h=height, and w=width
      sf = sf.view(n, h * w)
      nf = nf.view(n, h * w)

      # calculate gram matrices
      sf = gram_matrix(sf)
      nf = gram_matrix(nf)

      # compute style loss
      sty_loss += style_loss(nf, sf, n, h, w)

    # calculate total loss and then backpropagate
    total_loss = cont_loss + sty_loss * style_weights
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if( ((epoch + 1) % (n_epochs//10)) == 0):
      print('Current epoch is: ' + str(epoch))

    if( ((epoch + 1) % (n_epochs//4)) == 0):
      # save output image
      denormalize = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
      final_img = new_img.clone().squeeze().cpu()
      final_img = denormalize(final_img).clamp_(0, 1)
      torchvision.utils.save_image(final_img, './out_imgs/{}/{}_epoch_{}.png'.format(name, name, epoch))

"""
##############################################################################
# Parser
##############################################################################
"""
def create_parser():
  """
  return: parser inputs
  """
  parser = argparse.ArgumentParser(
        description='PyTorch Neural Art.')

  # function to allow parsing of true/false boolean
  def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')
    
  # arguments for logging
  parser.add_argument('--log_name', type=str, default=None,
                        help='Specify log name, if not specified then will default to args.name; default=None')

  # arguments for content and style image locations
  parser.add_argument('--cont_img_loc', type=str, default='https://ichef.bbci.co.uk/images/ic/720x405/p0517py6.jpg', # cute cat img default :3
                        help='Determine location of cont img, whether online or not; default=https://ichef.bbci.co.uk/images/ic/720x405/p0517py6.jpg')
  parser.add_argument('--cont_on', type=str2bool, nargs='?', default=True,
                        help='Flag to determine if content image is from online link or not; default=True')

  parser.add_argument('--style_img_loc', type=str, default='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg', # cute cat img default :3
                        help='Determine location of style img, whether online or not; default=https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
  parser.add_argument('--style_on', type=str2bool, nargs='?', default=True,
                        help='Flag to determine if style image is from online link or not; default=True')
  
  # arguments for image parameters
  parser.add_argument('--img_size', type=int, default=224,
                        help='Determines input image scale pixel x pixel size, then center crop of 224 for VGG19; default=224')
  parser.add_argument('--style_weights', type=int, default=1000000,
                        help='Determines how much effect style has on image; default=1000000')
  parser.add_argument('--name', type=str, default='test',
                        help='Determines output image name; default=test')

  # arguments for other hyperparameters
  parser.add_argument('--n_epochs', type=int, default=500,
                        help='Defines num epochs for training; default=500')
  parser.add_argument('--lr', type=float, default=0.03,
                        help='Defines learning rate for training; default=0.03')
  parser.add_argument('--random_seed', type=int, default=7,
                        help='Defines random seed value; default=7')

  args = parser.parse_args()

  return args


"""
##############################################################################
# Main, where all the magic starts~
##############################################################################
"""
def main(): 
  """
  Runs through two images iteratively to make neural artwork~
  """
  
  # parsing input arguments
  args = create_parser()

  # double check random seed
  if (args.random_seed == None):
       args.random_seed = random.randint(1, 1000)

  # set reproducible random seed
  np.random.seed(args.random_seed)
  torch.manual_seed(args.random_seed)
  
  # make logs directory if it doesn't exist
  if(os.path.exists('./logs/') == False):
    os.mkdir('logs')   
  # make output imgs directory if it doesn't exist
  if(os.path.exists('./out_imgs/') == False):
    os.mkdir('out_imgs')

  # make new folder inside img output based off args.name
  if(os.path.exists('./out_imgs/{}'.format(args.name)) == False):
    os.mkdir('./out_imgs/{}'.format(args.name))

  # if log name not specified then just set to name
  if(args.log_name == None):
    args.log_name = args.name

  # open log files and fill in hyperparameters
  today_date = str(datetime.today()).replace(':', '_').replace(' ', '_')
  log_file = open('./logs/log_run_{}_{}.txt'.format(today_date, args.log_name), 'w+')
  log_file.write('cont_img_loc   = {} \n'.format(args.cont_img_loc))
  log_file.write('style_img_loc  = {} \n'.format(args.style_img_loc))
  log_file.write('cont_on        = {} \n'.format(args.cont_on))
  log_file.write('style_on       = {} \n'.format(args.style_on))
  log_file.write('name           = {} \n'.format(args.name))
  log_file.write('img_size       = {:d} \n'.format(args.img_size))
  log_file.write('num_epochs     = {:d} \n'.format(args.n_epochs))
  log_file.write('learning_rate  = {} \n'.format(args.lr))
  log_file.write('random_seed    = {:d} \n'.format(args.random_seed))
  log_file.write('style_weights  = {:d} \n'.format(args.style_weights))

  # start timer
  start_time = time.time()

  # load images
  orig_cont, orig_style, cont_img, style_img, new_img = load_imgs(args.cont_img_loc, args.cont_on, args.style_img_loc, args.style_on, args.img_size)
  log_file.write('Images loaded successfully \n')
  print('Images loaded successfully')

  # display original images
  imshow(orig_cont, title='content', orig=True)
  imshow(orig_style, title='style', orig=True)
  #imshow(new_img, title='new')

  # define optimizer
  optimizer = torch.optim.Adam([new_img], lr=args.lr, betas=[0.5, 0.999])

  # import pretrained model
  model = Feature_Extraction().to(device).eval()
  log_file.write('Model loaded successfully \n')
  print('Model loaded successfully \n')

  # train networks
  log_file.write('Training commencing \n')
  print('Training commencing \n')
  training(model, optimizer, args.n_epochs, cont_img, style_img, new_img, args.style_weights, args.name)
  mem_allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
  log_file.write('Memory allocated was: ' + str(mem_allocated) + '\n')
  log_file.write('Training has finished \n')
  print('Training has finished \n')

  # end timer
  end_time = time.time()
  time_taken = end_time - start_time

  # write timing out to log file and terminal
  log_file.write('Time taken: {:.2f} mins \n'.format(time_taken/60))
  print('Time taken: {:.2f} mins \n'.format(time_taken/60))

  log_file.close()

  

"""
##############################################################################
# Main call
##############################################################################
"""
if __name__ == '__main__':
    main()
