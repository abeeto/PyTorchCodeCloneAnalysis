import numpy as np
from PIL import Image
import os
import h5py

def load_data(h5_dir=None, imgs_dir=None):
    '''
    h5_dir   -> str with the absolute of reletive path
                where the h5_file is stored.
    imgs_dir -> str with dir where the images are stored
    
    !!! h5 file should always be called 'data.h5' and
        data are stored in key 'data'.
        
    Returns out = None if the process failed.
    '''
    # Get parent directory of this file
    parent_path = os.path.dirname(os.path.realpath(__file__))
    
    out = None
    if 'data.h5' not in os.listdir(h5_dir):
        if imgs_dir is None:
            print('Could not find data.')
        else:
            create_h5_file(imgs_dir, h5_dir)
            with h5py.File('data.h5', 'r') as hf:
                out = hf['data'][:]
    else:
        goto = os.path.join(h5_dir, 'data.h5')
        with h5py.File(goto, 'r') as hf:
            out = hf['data'][:]
  
    return out
    
def create_h5_file(imgs_dir, store_at_dir = ''):
    '''
    imgs_dir     -> path of folder where the images are stored
    store_at_dir -> path of directory where data.h5 will be stored
    '''
    paths = get_img_paths(imgs_dir)
    color64 = create_img_arrays(paths)
    
    n = len(paths)
    if n > 1:
        store_path = os.path.join(store_at_dir, 'data.h5')
        with h5py.File(store_path, 'w') as hf:
            hf.create_dataset('data', data=color64)
        print('%d images where stored in ' %  + store_path)
    else:
        print('Images Found: %d. ' % n + 'Not enough to form h5 file')
    
def get_img_paths(path):
    '''
    Gets path of the directory where the images are stored.
    Returns list with the absolute paths of all .png files. 
    '''
    paths = []
    for i in os.listdir(path):
        if i.endswith('jpg') or i.endswith('.png'):
            p = os.path.join(path, i)
            paths.append(p)
   
    return paths

def create_img_arrays(paths):
    '''
    Inputs:
    list with the absolute paths of the images
    Returns:
    color64 -> array of size (obs,64,64,3) with 
               the RGB values of the images
    '''
    n = len(paths)
    color64 = np.zeros((n,64,64,3), dtype=np.uint8)
    for i in range(n):
        with Image.open(paths[i]) as img:
            img = img.resize((64,64), Image.ANTIALIAS)
            color64[i,:,:,:] = np.array(img)
            
    return color64