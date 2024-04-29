import numpy as np
import nibabel as nib
import glob
#import matplotlib.pyplot as plt
from tifffile import imsave
import os
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
from skimage.transform import resize
import random

'''
Adapted from
# https://youtu.be/oB35sV1npVI
'''

scaler = MinMaxScaler()

def resize_volume(img):
    # not using this. will resize inside the training algorithm, and make sure that masks are handled in a different
    # way to the images
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 128
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = resize(img, [desired_width, desired_height, desired_depth], order=0)
    #img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=3)
    return img


DATASET_PATH = 'D:/ctp_project_data/DWI_Training_Data/'

TRAIN_DATASET_PATH = DATASET_PATH + 'train/'
VAL_DATASET_PATH = DATASET_PATH + 'val/'

def makedata(DATASET_PATH, string):

    # cbf_list = sorted(glob.glob(DATASET_PATH + string + '/' + '*/*cbf.nii.gz'))
    # dt_list = sorted(glob.glob(DATASET_PATH + string + '/' + '*/*dt.nii.gz'))
    # cbv_list = sorted(glob.glob(DATASET_PATH + string + '/' + '*/*cbv.nii.gz'))
    # mtt_list = sorted(glob.glob(DATASET_PATH + string + '/' + '*/*mtt.nii.gz'))
    mask_list = glob.glob(DATASET_PATH + string + '/' + '*/*seg.nii.gz')
    dwi_list = glob.glob(DATASET_PATH + string + '/' + '*/*dwi.nii.gz')

    for img in range(len(mask_list)):  # Using t1_list as all lists are of same size
        print("Now preparing image and masks number: ", img)

        # temp_image_cbf = nib.load(cbf_list[img]).get_fdata()
        # temp_image_cbf = scaler.fit_transform(temp_image_cbf.reshape(-1, temp_image_cbf.shape[-1])).reshape(
        #     temp_image_cbf.shape)
        #
        # temp_image_dt = nib.load(dt_list[img]).get_fdata()
        # temp_image_dt = scaler.fit_transform(temp_image_dt.reshape(-1, temp_image_dt.shape[-1])).reshape(
        #     temp_image_dt.shape)
        #
        # temp_image_cbv = nib.load(cbv_list[img]).get_fdata()
        # temp_image_cbv = scaler.fit_transform(temp_image_cbv.reshape(-1, temp_image_cbv.shape[-1])).reshape(
        #     temp_image_cbv.shape)
        #
        # temp_image_mtt = nib.load(mtt_list[img]).get_fdata()
        # temp_image_mtt = scaler.fit_transform(temp_image_mtt.reshape(-1, temp_image_mtt.shape[-1])).reshape(
        #     temp_image_mtt.shape)
        temp_image_dwi = nib.load(dwi_list[img]).get_fdata()
        # print(temp_image_dwi.shape)
        # temp_image_dwi = scaler.fit_transform(temp_image_dwi.reshape(-1, temp_image_dwi.shape[-1])).reshape(
        #    temp_image_dwi.shape)
        temp_mask = nib.load(mask_list[img]).get_fdata()
        temp_mask = temp_mask.astype(np.uint8)

        # resize images
        # temp_image_cbf = resize_volume(temp_image_cbf)
        # temp_image_dt = resize_volume(temp_image_dt)
        # temp_image_cbv = resize_volume(temp_image_cbv)
        # TODO: keywords for mask interpolation
        #temp_image_dwi = resize_volume(temp_image_dwi)
        #temp_mask = resize_volume(temp_mask)

        #temp_combined_images = np.stack([temp_image_cbf, temp_image_dt, temp_image_cbv, temp_image_mtt], axis=3)
        print("size: {}".format(temp_mask.shape))
        val, counts = np.unique(temp_mask, return_counts=True)

        NEW_DATA = 'input_data_numpy/'
        if not os.path.exists(DATASET_PATH + NEW_DATA + string + '/images/'):
            os.makedirs(DATASET_PATH + NEW_DATA + string + '/images/')
        if not os.path.exists(DATASET_PATH + NEW_DATA + string + '/masks/'):
            os.makedirs(DATASET_PATH + NEW_DATA + string + '/masks/')
        np.save(DATASET_PATH + NEW_DATA + string + '/images/image_' + str(img) + '.npy', temp_image_dwi)
        np.save(DATASET_PATH + NEW_DATA + string + '/masks/mask_' + str(img) + '.npy', temp_mask)


makedata(DATASET_PATH, 'train')
# makedata(DATASET_PATH, 'train')
