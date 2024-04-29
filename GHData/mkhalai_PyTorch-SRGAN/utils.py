
import torch
from torchvision import transforms
from transforms import test_tfm, inverse_tfm
from config import *
import os
import random
from PIL import Image

#%%
CROP_SIZE = 500
files = [os.path.join(IMAGE_PATH, x) for x in os.listdir(IMAGE_PATH)]

def concat_images_h(lr, sr, hr):
    # print(type(lr),type(sr),type(hr))
    x = Image.new("RGB", (lr.width * 3, lr.height))
    x.paste(lr, (0, 0))
    x.paste(sr, (lr.width, 0))
    x.paste(hr, (lr.width * 2, 0))
    return x

random_crop = transforms.RandomCrop((CROP_SIZE, CROP_SIZE))

def save_image(generator, epoch):
    fpath = random.choice(files)  # path
    img = Image.open(fpath)  # pil

    hr = random_crop(img) #<500,500 PIL>
    lr = hr.resize((CROP_SIZE//4,CROP_SIZE//4),resample=Image.BICUBIC)
    lr_input = test_tfm(lr).unsqueeze(0) #shape=[1,3,125,125]
    lr = lr.resize((CROP_SIZE, CROP_SIZE), resample=Image.BICUBIC)

    generator.eval()
    
    with torch.no_grad():
        sr = generator(lr_input).to(device).squeeze(0)
        sr = inverse_tfm(sr)
        conc = concat_images_h(lr, sr, hr)
        conc.save(f"{IMAGE_SAVE_PATH}/_{epoch}.jpg")

    generator.train()


def load_checkpoint(model, optimizer, fpath, lr):
    checkpoint = torch.load(fpath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    
def save_checkpoint(model, optimizer, fpath):
    
    torch.save({
        'model_state_dict' : model.state_dict(),
        'opt_state_dict' : optimizer.state_dict()
    }, fpath)
   

#%%