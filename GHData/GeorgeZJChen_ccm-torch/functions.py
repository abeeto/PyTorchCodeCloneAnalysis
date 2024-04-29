from __future__ import print_function, division
import torch
from torchvision import models as torch_models
import numpy as np
from PIL import Image, ImageOps
import string
import os
import random
import time
from tqdm import tqdm
from scipy import io as scipy_io
import math
import pickle
import h5py
import cv2

def move_files(path_to_load, part='A'):
  if not path_to_load.endswith('/'):
    path_to_load += '/'
  train_ptl = path_to_load + 'train/'
  test_ptl = path_to_load + 'test/'

  if not os.path.exists(train_ptl):
    os.makedirs(train_ptl)
  if not os.path.exists(test_ptl):
    os.makedirs(test_ptl)
  for _, _, files in os.walk("./shanghaitech/part_"+part+"_final/train_data/ground_truth"):
    for filename in files:
      if '.mat' in filename:
        new_name = filename.replace('GT_','')
        os.rename("./shanghaitech/part_"+part+"_final/train_data/ground_truth/"+filename, train_ptl + new_name)
        os.rename("./shanghaitech/part_"+part+"_final/train_data/images/"+new_name.replace('.mat','.jpg'), train_ptl + new_name.replace('.mat','.jpg'))
  for _, _, files in os.walk("./shanghaitech/part_"+part+"_final/test_data/ground_truth"):
    for filename in files:
      if '.mat' in filename:
        new_name = filename.replace('GT_','')
        os.rename("./shanghaitech/part_"+part+"_final/test_data/ground_truth/"+filename, test_ptl + new_name)
        os.rename("./shanghaitech/part_"+part+"_final/test_data/images/"+new_name.replace('.mat','.jpg'), test_ptl + new_name.replace('.mat','.jpg'))

def load_data_names(train=True, part='A'):
  names = []
  if train:
    for _, _, files in os.walk('./datasets/shanghaitech/'+part+'/train/'):
      for filename in files:
        if '.mat' in filename:
          names.append(filename.replace('.mat',''))
  else:
    pass
    for _, _, files in os.walk('./datasets/shanghaitech/'+part+'/test'):
        for filename in files:
          if '.jpg' in filename:
            names.append(filename.replace('.jpg',''))
  return names
def load_data_ShanghaiTech(path):
  img = Image.open(path+'.jpg')
  coords = scipy_io.loadmat(path+'.mat')['image_info'][0][0][0][0][0]
  return img, coords
def display_set_of_imgs(images, rows=2, size=0.5, name='0'):
  n_images = len(images)
  with open('./output/images/'+str(name)+'-'+id_generator(5)+'.pkl', 'wb') as f:
      pickle.dump(images, f)
  # fig = plt.figure()
  # plt.axis('off')
  # for n, image in enumerate(images):
  #     if image.shape[-1] == 1:
  #       image = np.reshape(image, (image.shape[0], image.shape[1]))
  #       a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
  #       a.axis('off')
  #       a.set_title(str(image.shape)+', '+str(round(np.sum(image), 2)))
  #       plt.imshow(image, cmap=plt.get_cmap('jet'))
  #     elif  image.shape[-1] == 3:
  #       a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
  #       a.axis('off')
  #       a.set_title(str(image.shape))
  #       plt.imshow(image)
  # fig.set_size_inches(np.array(fig.get_size_inches()) * size)
  # plt.show()
def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in range(size))
def total_parameters(net):
  model_parameters = filter(lambda p: p.requires_grad, net.parameters())
  total_parameters = sum([np.prod(p.size()) for p in model_parameters])
  return total_parameters
def gaussian_kernel(shape=(32,32),sigma=5):
  """
  2D gaussian kernel which is equal to MATLAB's
  fspecial('gaussian',[shape],[sigma])
  """
  radius_x,radius_y = [(radius-1.)/2. for radius in shape]
  y_range,x_range = np.ogrid[-radius_y:radius_y+1,-radius_x:radius_x+1]
  h = np.exp( -(x_range*x_range + y_range*y_range) / (2.*sigma*sigma) )

  # finfo(dtype).eps: a very small value
  h[ h < np.finfo(h.dtype).eps*h.max()] = 0
  sumofh = h.sum()
  if sumofh != 0:
      h /= sumofh
  return h
def get_downsized_density_maps(density_map):
  ddmaps = []
  ratios = [8,16,32,64,128]
  ddmap = torch.nn.functional.avg_pool2d(density_map, ratios[0], ratios[0], padding=0) * (ratios[0] * ratios[0])
  ddmaps.append(torch.squeeze(ddmap,0))
  if len(ratios)>1:
    for i in range(len(ratios)-1):
      ratio = int(ratios[i+1]/ratios[i])
      ddmap = torch.nn.functional.avg_pool2d(ddmap, ratio, stride=ratio, padding=0) * (ratio * ratio)
      ddmaps.append(torch.squeeze(ddmap,0))
  return ddmaps, [torch.flip(ddmap, [len(ddmap.size())-1]) for ddmap in ddmaps]
def random_size(rate_range=[1.1, 1.6], input_size=[384, 512], img_size=[None,None]):
  img_height, img_width = img_size
  input_height, input_width = input_size
  resized_height = img_height
  resized_width = img_width
  erate = rate_range[0] + (rate_range[1]-rate_range[0])*random.random()
  if img_height <= input_height*erate:
    resized_height = int(input_height*erate)
    resized_width = resized_height/img_height*img_width
    if resized_width <= input_width*erate:
      resized_width = int(input_width*erate)
      resized_height = resized_width/img_width*img_height
  elif  img_width <= input_width*erate:
    resized_width = int(input_width*erate)
    resized_height = resized_width/img_width*img_height
    if resized_height <= input_height*erate:
      resized_height = int(input_height*erate)
      resized_width = resized_height/img_height*img_width
  return int(resized_height), int(resized_width)
def fit_grid(img_height, img_width, input_size=[384,512]):
  input_height, input_width = input_size
  columns = max(1, int(round(img_width/input_width)))
  rows = max(1, int(round(input_width*columns*img_height/img_width/input_height)))
  return rows, columns
def get_coords_map(coords, resize, img_size):
  resized_height, resized_width = resize
  img_height, img_width = img_size
  new_coords = []
  for coord in coords:
    new_coord = [0,0]
    new_coord[0] = min(coord[0], img_width-1)*resized_width/img_width
    new_coord[1] = min(coord[1], img_height-1)*resized_height/img_height
    new_coords.append(new_coord)
  coords_map = np.zeros([1, 1, resized_height, resized_width])
  for coord in new_coords:
    coords_map[0][0][int(coord[1])][int(coord[0])] += 1
  return coords_map

def preprocess_data(names, data_path, save_path='./processed', random_crop=None, input_size=[384, 512]
                    , test=False, test_dict=None):
  if not data_path.endswith('/'):
    data_path += '/'
  if not save_path.endswith('/'):
    save_path += '/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  if test and not 'names_to_name' in test_dict:
    test_dict['names_to_name'] = {}

  input_height, input_width = input_size
  prog = 0
  out_names = []
  kernel_size = 49

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  class GetDensityMap(torch.nn.Module):
    def __init__(self):
      super(GetDensityMap, self).__init__()
      kernel = gaussian_kernel(shape=(kernel_size,kernel_size),sigma=10)
      kernel = np.reshape(kernel, (1,1)+kernel.shape)
      self.kernel = torch.from_numpy(kernel).float().to(device)
      self.padding = int((kernel_size-1)/2)
    def forward(self, coords_map):
      return torch.nn.functional.conv2d(coords_map, self.kernel, padding=self.padding)
  class GetDownsizeDensityMaps(torch.nn.Module):
    def __init__(self):
      super(GetDownsizeDensityMaps, self).__init__()
    def forward(self, density_map):
      return get_downsized_density_maps(density_map)

  getDensityMap = GetDensityMap().to(device)
  getDownsizeDensityMaps = GetDownsizeDensityMaps().to(device)

  for ni in tqdm(range(len(names))):
    name = data_path +  names[ni]

    img, coords = load_data_ShanghaiTech(name)

    if img.mode !='RGB':
      img = img.convert('RGB')
    img_width, img_height = img.size

    imgs = []
    dmaps = []

    rows, columns = fit_grid(img_height, img_width, input_size=[input_height, input_width])

    resized_height = rows*input_height
    resized_width = columns*input_width
    new_img = img.resize((resized_width, resized_height))
    coords_map = get_coords_map(coords, resize=[resized_height, resized_width], img_size=[img_height, img_width])
    coords_map = torch.from_numpy(coords_map).float().to(device)
    dmap = getDensityMap(coords_map)
    for row in range(rows):
      for col in range(columns):
        crop_top = input_height*row
        crop_left = input_width*col
        crop_bottom = crop_top + input_height
        crop_right = crop_left + input_width
        img_crop = new_img.crop((crop_left, crop_top, crop_right, crop_bottom))

        ddmaps, ddmaps_mirrored = getDownsizeDensityMaps(dmap[:, :, crop_top:crop_bottom, crop_left:crop_right])

        imgs.append(img_crop)
        dmaps.append(to_np(ddmaps))
        if not test:
          imgs.append(ImageOps.mirror(img_crop))
          dmaps.append(to_np(ddmaps_mirrored))

    if random_crop is not None and not (rows==1 and columns==1) and not test:
      for b in range(random_crop):

        crop_top = 0 if rows==1 else np.random.randint(0, resized_height - input_height)
        crop_left = 0 if columns==1 else np.random.randint(0, resized_width - input_width)
        crop_bottom = crop_top + input_height
        crop_right = crop_left + input_width
        img_crop = new_img.crop((crop_left, crop_top, crop_right, crop_bottom))

        density_map_ = dmap[:, :, crop_top:crop_bottom, crop_left:crop_right]
        ddmaps, ddmaps_mirrored = getDownsizeDensityMaps(density_map_)

        imgs.append(img_crop)
        dmaps.append(to_np(ddmaps))

        imgs.append(ImageOps.mirror(img_crop))
        dmaps.append(to_np(ddmaps_mirrored))

    for i in range(len(imgs)):
      new_name = id_generator()

      img_i = imgs[i]
      if not test and random.random()>0.9:
        img_i = img_i.convert('L').convert('RGB')
      img_i.save(save_path + new_name + '.jpg', 'JPEG')
      with open(save_path + new_name + '.pkl', 'wb') as f:
        pickle.dump(dmaps[i], f)

      out_names.append(save_path + new_name)

      if test:
        test_dict['names_to_name'][save_path + new_name] = name
    if test:
      test_dict[name] = {
          'predict': -1,
          'truth': len(coords)
      }
  return out_names

def set_pretrained(net):

  vgg16 = torch_models.vgg16(pretrained=True)
  vgg_dict = vgg16.state_dict()

  torch_dict = net.state_dict()

  vgg_p_ids = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]
  torch_p_ids = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12]
  for i in range(len(vgg_p_ids)):
    torch_name_w = 'vgg.'+str(torch_p_ids[i])+'.0.weight'
    torch_name_b = 'vgg.'+str(torch_p_ids[i])+'.0.bias'

    vgg_name_w = 'features.'+str(vgg_p_ids[i])+'.weight'
    vgg_name_b = 'features.'+str(vgg_p_ids[i])+'.bias'

    assert torch_name_w in torch_dict
    assert torch_name_b in torch_dict
    torch_dict[torch_name_w] = vgg_dict[vgg_name_w]
    torch_dict[torch_name_b] = vgg_dict[vgg_name_b]
  net.load_state_dict(torch_dict)
#   test_set_pretrained('vgg.'+str(torch_p_ids[0])+'.0.weight', 'features.'+str(vgg_p_ids[0])+'.weight'
#                       , net, vgg16)

def test_set_pretrained(torch_name, vgg_name, torch_net, vgg16):
  def check_equal(a, b):
    a = a.flatten()
    b = b.flatten()
    if len(a) != len(b):
        print('inequivalent length:', len(a), '!=', len(b))
        return False
    for m in range(len(a)):
        if abs(a[m]-b[m]) > 0.000001:
            print(a[m], '!=', b[m], 'at', m)
            return False
    return True
  vgg_dict = vgg16.state_dict()
  torch_dict = torch_net.state_dict()
  vgg_data = vgg_dict[vgg_name].data.numpy()
  torch_data = torch_dict[torch_name].cpu().data.numpy()
  assert check_equal(vgg_data, torch_data)
def moving_average(new_val, last_avg, theta=0.95):
  return round((1-theta) * new_val + theta* last_avg, 2)
def moving_average_array(new_vals, last_avgs, theta=0.95):
  return [round((1-theta) * new_vals[i] + theta* last_avgs[i], 2) for i in range(len(new_vals))]
def MAE(predicts, targets):
  return round( np.mean( np.absolute( np.sum(predicts, (1,2,3)) - np.sum(targets, (1,2,3)) )), 1)
def normalize(imgs):
  new_imgs = []
  for i in range(len(imgs)):
    img = imgs[i] / 255
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    new_imgs.append(img)
  return new_imgs
def denormalize(img):
  img *= [0.229, 0.224, 0.225]
  img += [0.485, 0.456, 0.406]
  img *= 255
  return img.astype('uint8')
def to_np(input):
  if type(input) is list:
    return [ tensor.cpu().detach().numpy() for tensor in input ]
  return input.cpu().detach().numpy()
def next_batch(batch_size, names):
  b = np.random.randint(0, len(names), [batch_size])
  _names = names[b]

  imgs = []
  targets15 = []
  targets14 = []
  targets13 = []
  targets12 = []
  targets11 = []
  targets10 = []

  for name in _names:
    imgs.append(np.asarray(Image.open(name+'.jpg')))
    target10, target11, target12, target13, target14 = pickle.load(open(name+'.pkl','rb'))
    targets15.append(np.reshape(np.sum(target14), [1,1,1]))
    targets14.append(target14)
    targets13.append(target13)
    targets12.append(target12)
    targets11.append(target11)
    targets10.append(target10)

  targets = [targets15, targets14, targets13, targets12, targets11, targets10]
  targets = [np.array(target) for target in targets]
  inputs = np.array(normalize(imgs))
  return inputs, targets
