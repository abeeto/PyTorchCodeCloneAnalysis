import random
import os
img_root = '/home/gwl/datasets/DUT-OMRON/DATASET/images'
gt_root = '/home/gwl/datasets/DUT-OMRON/DATASET/masks'
val_imgroot = '/home/gwl/datasets/DUT-OMRON/val/images'
val_gtroot = '/home/gwl/datasets/DUT-OMRON/val/gt'
file_names = os.listdir(img_root)
img_names = []
gt_names = []
names = []
for i, name in enumerate(file_names):
    if not name.endswith('.jpg'):
        continue
    img_names.append(
        os.path.join(img_root, name[:-4] + '.jpg')
    )
    names.append(name[:-4])

slice = random.sample(names,1793)
src_img = []
dst_img = []
src_gt = []
dst_gt = []
for i,name in enumerate(slice):
    src_img.append(os.path.join(img_root,name+'.jpg'))
    dst_img.append(os.path.join(val_imgroot,name+'.jpg'))
    src_gt.append(os.path.join(gt_root,name+'.png'))
    dst_gt.append(os.path.join(val_gtroot,name+'.png'))
    os.rename(src_img[i],dst_img[i])
    os.rename(src_gt[i],dst_gt[i])
