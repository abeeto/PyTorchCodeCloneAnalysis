
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
import os
from PIL import Image
from skimage import io, transform, color
import torch
from tqdm import tqdm
from pathlib import Path
from torchvision.utils import save_image

class UpsampleDataset(Dataset):
    def __init__(self, path, i_image_size, o_image_size, is_inplace):
        self.root_dir = path
        
        self.i_image_size = i_image_size
        self.o_image_size = o_image_size

        self.x = []
        self.y = []

        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.root_dir) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]

        self.is_inplace = is_inplace


    def transform_dataset(self, out):
        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.root_dir) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]
        i_transform = T.Resize((self.i_image_size, self.i_image_size))
        o_transform = T.Resize((self.o_image_size, self.o_image_size))
        
        
        for idx, img_name in tqdm(enumerate(self.images)):
            image = Image.open(img_name).convert('RGB')
            i_image = i_transform(image)
            o_image = o_transform(image)

            
            i_image.save(os.path.join(out, f'x/{idx}.jpg'))
            o_image.save(os.path.join(out, f'y/{idx}.jpg'))



    def gpu_precache(self, device):
        pass

        x_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(self.root_dir, 'x')) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]
        y_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(self.root_dir, 'y')) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]

        self.len = min(len(x_images), len(y_images))
        print(f'Got {self.len} images')

    def __len__(self):
        return len(self.images) if self.is_inplace else self.len
    
    def __getitem__(self, idx):

        i_transform = T.Resize((self.i_image_size, self.i_image_size))
        o_transform = T.Resize((self.o_image_size, self.o_image_size))
        to_tensor = T.ToTensor()

        if self.is_inplace:
            image = Image.open(self.images[idx]).convert('RGB')

            i_image = o_transform(i_transform(image))
            o_image = o_transform(image)
        else:
            i_image = Image.open(os.path.join(self.root_dir, 'x', f'{idx}.jpg')).convert('RGB')
            o_image = Image.open(os.path.join(self.root_dir, 'y', f'{idx}.jpg')).convert('RGB')

        # i_image = i_transform(to_tensor(i_image))
        # o_image = o_transform(to_tensor(o_image))
        return to_tensor(i_image), to_tensor(o_image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-p', action='store', dest='path')
    parser.add_argument('-o', action='store', dest='out')

    args = parser.parse_args()


    # transform all images to resized
    Path(os.path.join(args.out, 'x')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.out, 'y')).mkdir(parents=True, exist_ok=True)
    
    UpsampleDataset(args.path, 64, 100).transform_dataset(args.out)
