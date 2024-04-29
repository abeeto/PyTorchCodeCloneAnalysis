import glob
import numpy as np
from keras.preprocessing.image import load_img,img_to_array

#image  array size
img_size = (640,360)
#load images Folder
dir_name = 'mov'
#File type
file_type  = 'jpg'

#load images and image to array
img_list = glob.glob('./' + dir_name + '/*.' + file_type)

temp_img_array_list = []

for img in img_list:
    temp_img = load_img(img,grayscale=False,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    temp_img_array_list.append(temp_img_array)

temp_img_array_list = np.array(temp_img_array_list)

#save np.array
np.save(dir_name+'.npy',temp_img_array_list)
