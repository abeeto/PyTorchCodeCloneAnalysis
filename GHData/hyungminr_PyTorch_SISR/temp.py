import os
import shutil
import glob

binD = glob.glob('./data/benchmark/REDS/bin/train/train_blur_bicubic/*/*/*.pt')
binE = glob.glob('./data/benchmark/REDS/bin_E/train/train_sharp/*/*.pt')

from tqdm import tqdm

for b in tqdm(binD):
    new_b = b.replace('/bin/', '/bin_E/')
    new_dir = '/'.join(new_b.split('/')[:-1])
    os.makedirs(new_dir, exist_ok=True)
    shutil.move(b, new_b)
