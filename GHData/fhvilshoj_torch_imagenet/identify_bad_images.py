import os
from PIL import Image
import numpy as np
from tqdm import tqdm

files = os.listdir('images')
files = [os.path.join('images', f) for f in files if 'random' not in f]

errors = []
for f in tqdm(files):
    img = np.array(Image.open(f))
    if (not len(img.shape)>2) or img.shape[-1] != 3:
        # Need all images to have three channels exactly
        errors.append(f)

import pickle 
with open('bad_images.pkl', 'wb') as f:
    pickle.dump(errors, f)


