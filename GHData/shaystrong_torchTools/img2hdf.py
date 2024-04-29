import h5py
import glob
import os.path
import time
from PIL import Image
import tempfile
import numpy as np

ims = [np.array(Image.open(f)) for f in glob.glob('*.jpg')]

def setup_hdf5_file(shape, name, **options):
    f = h5py.File(name+'.h5', 'w')
    f.create_dataset('data', shape, **options)
    f.create_dataset('labels', (len(ims),), **options)
    return f

def save_images(f):
    for i, im in enumerate(ims):
        f['data'][i,...] = im.T
        f['labels'][i] = int(1)
    f.close()

def benchmark(name, **options):
    tstart = time.time()
    f = setup_hdf5_file((len(ims),3,256, 256), name, **options)
    save_images(f)
    tstop = time.time()
    size = os.path.getsize(name+'.h5')
    print("{0}: {1:.1f}s, {2}MB".format(name, tstop-tstart, size//1e6))


benchmark('data4torch')
