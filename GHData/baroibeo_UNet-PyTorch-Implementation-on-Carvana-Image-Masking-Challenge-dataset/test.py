import torch
import os
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import cv2
from albumentations.pytorch import ToTensorV2
from model import UNet
from utils import load_checkpoint,get_transform
from PIL import Image

def combine(img,mask):
    """
    img:PIL image RGB
    mask : PIL image gray
    combine mask into img
    """
    mask=np.array(mask)
    height,width=(mask).shape
    mask_rgb=np.zeros((height,width,3),dtype=np.uint8)
    img=np.array(img)
    rows,cols=np.where(mask==255.0)
    mask_rgb[rows,cols]=(255,0,150)
    mask_rgb[mask==0.0]=(0,255,0)
    return cv2.addWeighted(img,0.8,mask_rgb,0.3,0)

if __name__=="__main__":
    resized_height=128
    resized_width=128
    num_test=10
    test_dir="D:/datasets/carvana_image_masking_challenge/test/"
    test_img_names=os.listdir(test_dir)
    test_result_masks="D:/U-Net PyTorch Implementation/test_results/masks/"
    test_imgs="D:/U-Net PyTorch Implementation/test_results/imgs/"
    test_combine="D:/U-Net PyTorch Implementation/test_results/combinations/"
    model=UNet(in_channels=3,out_channels=1)
    checkpoint=torch.load("my_checkpoint.pth.tar",map_location="cpu")
    load_checkpoint(checkpoint,model)
    random_idx=np.random.choice(len(test_img_names),num_test)

    #Test transform
    transform=get_transform(resized_height,resized_width,train=False)
    
    model.eval()
    with torch.no_grad():
        for i in random_idx:
            img=Image.open(os.path.join(test_dir,test_img_names[i]))
            img_arr=np.array(img)
            img_tensor=transform(image=img_arr)['image']
            img_tensor=img_tensor.unsqueeze(0)
            out=model(img_tensor)
            out=(out>0.5).float()
            out=out.squeeze(0).squeeze(0).detach().numpy()
            out[out==1.0]=255.0
            out=Image.fromarray(out.astype(np.uint8))
            img=img.resize((resized_height,resized_width))
            img.save(test_imgs+str(i)+".jpg")
            out.save(test_result_masks+str(i)+".jpg")
            comb=combine(img,out)
            comb=Image.fromarray(comb.astype(np.uint8))
            comb.save(test_combine+str(i)+".jpg")

        