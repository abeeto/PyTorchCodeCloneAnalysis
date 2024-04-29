from collections import OrderedDict
from PIL import Image

import torch
from data import Cityscapes as dataset
from torch.autograd import Variable
from deeplab_multi import Encoder,Decoder

import torchvision.transforms as transforms

import transforms as ext_transforms
from args import get_arguments
import utils
from data.utils import get_files

args = get_arguments()
device = torch.device(args.device)   
# Get the color-encoding from the training dataset 
"""
image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])
label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        ext_transforms.PILToLongTensor()
    ])

train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)

class_encoding = train_set.color_encoding
"""
# Directly define the class-encoding without importing from the dataloader
class_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32))
    ])
num_classes = len(class_encoding)

# Set the root directory for the test set
test_folder = args.test_dir
img_extension = '.png'
Dataset = get_files(test_folder,extension_filter=img_extension)

# Define the segmentation model and load the trained weights
input_encoder = Encoder().to(device)
decoder_t=Decoder( num_classes).to(device)
model_path='save/deeplabv2'

if (args.device == 'cuda'):
    #cuda mode
    checkpoint = torch.load(model_path)
else:
    # cpu mode
    checkpoint = torch.load(model_path,map_location='cpu')
input_encoder.load_state_dict(checkpoint['encoder_i_state_dict'])
decoder_t.load_state_dict(checkpoint['decoder_t__state_dict'])

# Set the parameters for the input images
image_transform = transforms.Compose(
        [transforms.Resize((256, 512)),
         transforms.ToTensor()])

unloader = transforms.ToPILImage()
for i in range(len(Dataset)):
    data_path = Dataset[i]
    inputs = Image.open(data_path)
    inputs=image_transform(inputs)
    imname = Dataset[i].split('\\')[-1]
    print(imname)
    imname=args.test_out+'/'+imname
    inputs = Variable(inputs.to(device))
    
    # Give the pridiction outputs
    f_i=input_encoder(inputs.unsqueeze(0))
    outputs=decoder_t(f_i)

    # Transfer the probability maps to color labels maps
    _, predictions = torch.max(outputs, 1)

    label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()])
    
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    
    # Save the outputs as the color labels maps
    image = color_predictions.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(imname)

      