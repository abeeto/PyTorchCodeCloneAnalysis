from scipy.io import loadmat
import os
from shutil import copyfile

dataset_root = os.path.join('data', 'sdd')

def copy_file(src, dst):
  dst_folders = dst.split('/')[:-1]
  f = ''
  for i in range(len(dst_folders)):
    f = os.path.join(f, dst_folders[i])
    if not os.path.exists(f):
      os.mkdir(f)
  copyfile(src, dst)

def process(mode):
  file_list = os.path.join(dataset_root, mode+'_list.mat')     
  mat = loadmat(file_list)
  img_list = [l_[0][0] for l_ in mat['file_list']]
  for l_ in img_list:
    src = os.path.join(dataset_root, 'Images', l_)
    dst = os.path.join(dataset_root, mode, l_)
    copy_file(src, dst)

if __name__=='__main__':
  process('train')
  process('test')

