# -*- coding: utf-8 -*-

import torch
from Miniproject_2 import model

def psnr(denoised, ground_truth):
    #Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
    mse = torch.mean((denoised - ground_truth)** 2)
    psnr = -10 * torch . log10 ( mse + 10** -8)
    return psnr.item()

##############################################################
subset_train = 1000
subset_test = 10000

noisy_imgs_1 , noisy_imgs_2 = torch.load('train_data.pkl')

noisy_imgs_1 = noisy_imgs_1[0:subset_train,:,:,:]
noisy_imgs_2 = noisy_imgs_2[0:subset_train,:,:,:]

test_imgs , clean_imgs = torch.load ('val_data.pkl')
test_imgs = test_imgs[0:subset_test,:,:,:]
clean_imgs = clean_imgs[0:subset_test,:,:,:]
###############################################################



noise2noise = model.Model()

noise2noise.train(noisy_imgs_1, noisy_imgs_2, num_epochs=10)   

denoised = noise2noise.predict(test_imgs)
psnr_ = psnr(denoised/255, clean_imgs/255) 
print("Psnr for trained model is: "+str(psnr_))

loaded_model =  model.Model()
loaded_model.load_pretrained_model()

denoised = loaded_model.predict(test_imgs)
psnr_ = psnr(denoised/255, clean_imgs/255) 
print("Psnr for pretrained model is: "+str(psnr_))
