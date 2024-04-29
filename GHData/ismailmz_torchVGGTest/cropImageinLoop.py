#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from PIL import Image


savedir = "E:/Cropped/OK_64x64"
filename = "E:/Cropped/KimonoFrame17.bmp"
img = Image.open(filename)
width, height = img.size

start_pos = start_x, start_y = (0,0)  
cropped_image_size = w, h = (64, 64)         

frame_num = 1
for col_i in range (0, width, w):
    for row_i in range (0, height, h):
        crop = img.crop((col_i, row_i, col_i + w, row_i + h))
        save_to= os.path.join(savedir, "counter_{:03}.bmp")
        crop.save(save_to.format(frame_num))
        frame_num += 1

print("Done to crop images to 64x64")

savedir = "E:/Cropped/OK_32x32"
start_pos = start_x, start_y = (0,0)  
cropped_image_size = w, h = (32, 32)         

frame_num = 1
for col_i in range (0, width, w):
    for row_i in range (0, height, h):
        crop = img.crop((col_i, row_i, col_i + w, row_i + h))
        save_to= os.path.join(savedir, "counter_{:03}.bmp")
        crop.save(save_to.format(frame_num))
        frame_num += 1

print("Done to crop images to 32x32")

savedir = "E:/Cropped/OK_16x16"
start_pos = start_x, start_y = (0,0)  
cropped_image_size = w, h = (16, 16)         

frame_num = 1
for col_i in range (0, width, w):
    for row_i in range (0, height, h):
        crop = img.crop((col_i, row_i, col_i + w, row_i + h))
        save_to= os.path.join(savedir, "counter_{:03}.bmp")
        crop.save(save_to.format(frame_num))
        frame_num += 1

print("Done to crop images to 16x16")

savedir = "E:/Cropped/OK_8x8"
start_pos = start_x, start_y = (0,0)  
cropped_image_size = w, h = (8, 8)         

frame_num = 1
for col_i in range (0, width, w):
    for row_i in range (0, height, h):
        crop = img.crop((col_i, row_i, col_i + w, row_i + h))
        save_to= os.path.join(savedir, "counter_{:03}.bmp")
        crop.save(save_to.format(frame_num))
        frame_num += 1

print("Done to crop images to 8x8")

savedir = "E:/Cropped/OK_4x4"
start_pos = start_x, start_y = (0,0)  
cropped_image_size = w, h = (4, 4)         

frame_num = 1
for col_i in range (0, width, w):
    for row_i in range (0, height, h):
        crop = img.crop((col_i, row_i, col_i + w, row_i + h))
        save_to= os.path.join(savedir, "counter_{:03}.bmp")
        crop.save(save_to.format(frame_num))
        frame_num += 1

print("Done to crop images to 4x4")


# In[ ]:




