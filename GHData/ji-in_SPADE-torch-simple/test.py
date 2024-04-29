import argparse
from nets import *
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision import utils
from dataset import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoint/checkpoint_38epoch.pt",
        help="path to the model checkpoint",
    )

    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')

    parser.add_argument('--label_nc', type=int, default=19, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')

    parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')

    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

    parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')

    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')

    parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

    parser.add_argument('--norm_G', type=str, default='spectralspadeinstance3x3', help='instance normalization or batch normalization')

    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    parser.add_argument(
        "--pics", type=int, default=1, help="number of images to be generated"
    )

    parser.add_argument('--dataset_path', type=str, default='./dataset/spade_celebA_test', help='The path of dataset')

    parser.add_argument('--img_height', type=int, default=256, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256, help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--segmap_ch', type=int, default=3, help='The size of segmap channel')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    test_dataloader = load_dataset(args)
    
    netG = Generator(args).to(device).train()

    checkpoint = torch.load(args.ckpt)

    netG.load_state_dict(checkpoint["netG"])

    with torch.no_grad():
        for idx, (real_x, label) in tqdm(enumerate(test_dataloader)):
            label = label.long()
            real_x, label = real_x.to(device), label.to(device)
            
            # create one-hot label map
            label_map = label
            bs, _, h, w = label_map.size()
            nc = args.label_nc + 1 if args.contain_dontcare_label else args.label_nc
            FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            input_label = FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
            
            netG.eval()            
            sample = netG(input_semantics, z=None)
            
            utils.save_image(
                    sample,
                    f"test_image/test_{idx}.png",
                    nrow=1,
                )
