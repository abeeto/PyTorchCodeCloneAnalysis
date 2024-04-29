import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000   
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)

    img = np.asarray(img)
    img = img[:,:,:3]
    img = Image.fromarray(img)

    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def load_nii(filename,slc=None,slc_dim=None):
    nii = np.array(nib.load(filename).get_data())

    if slc != None:
        if slc_dim == None:
            print('Please specify dimension.')
            return
        elif slc_dim == 0:
            niarray = nii[slc,:,:]
        elif slc_dim == 1:
            niarray = nii[:,slc,:]
        elif slc_dim == 2:
            niarray = nii[:,:,slc]
    elif slc == None:
        if slc_dim != None:
            print('Please specify slc window.')
            return
        else:
            niarray = nii

    scale = MinMaxScaler(feature_range=(0,255))
    niarray = scale.fit_transform(niarray)
    niarray = niarray/255

    return niarray

    

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2) # swapped ch and w*h, transpose share storage with original
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # new_tensor for same dimension of tensor
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0) # back to tensor within 0, 1
    return (batch - mean) / std