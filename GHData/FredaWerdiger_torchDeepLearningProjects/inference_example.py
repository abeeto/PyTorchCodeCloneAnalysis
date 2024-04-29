# Imports
import os.path
import pathlib
import napari
import numpy as np
import torch
from skimage.transform import resize
from inference import predict
from transformations import normalize_01, normalize_01_clip99, re_normalize
from unet import UNet
import nibabel as nb
from sklearn.metrics import f1_score
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


if os.path.exists('D:'):
    root = pathlib.Path('D:/ctp_project_data/DWI_Training_Data_INSP/test/')
    ctp_df = pd.read_csv(
        'C:/Users/fwerdiger/PycharmProjects/study_design/study_lists/dwi_inspire_dl.csv',
        index_col='dl_id'
    )
elif os.path.exists('/media/'):
    root = '/media/mbcneuro/HDD1/DWI_Training_Data_INSP/test/'
    ctp_df = pd.read_csv(
        '/mbcneuro/PycharmProjects/study_design/study_lists/dwi_inspire_dl.csv',
        index_col='dl_id'
    )


# root directory
images_names = get_filenames_of_path(root / 'images')
targets_names = get_filenames_of_path(root / 'masks')

# read images and store them in memory
images = [nb.load(img_name).get_fdata() for img_name in images_names]
targets = [nb.load(tar_name).get_fdata() for tar_name in targets_names]

# Resize images and targets
images_res = [resize(img, (128, 128, 128)) for img in images]
resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
targets_res = [resize(tar, (128, 128, 128), **resize_kwargs) for tar in targets]

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model
device = 'cpu'
model = UNet(in_channels=1,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3).to(device)


model_name = 'dwi_model_10epoch_diceloss_lr0.01_clip199.pt'
model_weights = torch.load(pathlib.Path.cwd() / model_name)

model.load_state_dict(model_weights)

# preprocess function
def preprocess(img: np.ndarray):
    #img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, N, H, W]
    # have to make sure the transforms are in the right order cos normalise expects [C, N, H W]
    img = normalize_01_clip99(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add channel dimension [B, C, N, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# postprocess function
def postprocess(img: torch.tensor):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    img = re_normalize(img)  # scale it to the range [0-255]
    return img

# predict the segmentation maps
output = [predict(img, model, preprocess, postprocess, device) for img in images_res]
#
# viewer = napari.Viewer()
#
# idx = 59
# #while idx < (len(images_res) - 1):
# img_nap = viewer.add_image(images_res[idx], name='Input')
# tar_nap = viewer.add_labels(targets_res[idx].astype('int'), name='Target')
# out_nap = viewer.add_labels(output[idx], name='Prediction')
#  #   idx+=1
#
# import napari
#
# viewer = napari.Viewer()
#
# idx = 3
# img_nap = viewer.add_image(images_res[idx], name='Input')
# tar_nap = viewer.add_labels(targets_res[idx].astype('int'), name='Target')
# out_nap = viewer.add_labels(output[idx], name='Prediction')

def define_dvalues(dwi_img):
    dwi_img_small = dwi_img[10:120]
    steps = int(dwi_img_small.shape[0]/18)
    rem = int(dwi_img_small.shape[0]/steps)-18

    if rem % 2 == 0:
        d_min = 0 + int(rem/2*steps) + 1
        d_max = dwi_img_small.shape[0] - int(rem/2*steps)

    elif rem % 2 != 0:
        d_min = 0 + math.ceil(rem*steps/2)
        d_max = dwi_img_small.shape[0] - math.ceil(rem/2*steps)

    d = range(d_min + 10, d_max + 10, steps)

    if len(d) == 19:
        d = range(d_min + steps + 10, d_max + 10, steps)
    return d


def create_mrlesion_img(dwi_img, dwi_lesion_img, savefile, d, ext='png', dpi=250):
    dwi_lesion_img = np.rot90(dwi_lesion_img)
    dwi_img = np.rot90(dwi_img)
    mask = dwi_lesion_img < 1
    masked_im = np.ma.array(dwi_img, mask=~mask)

    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.4, wspace=0)

    axs = axs.ravel()

    for i in range(len(d)):
        axs[i].imshow(dwi_lesion_img[:, :,d[i]], cmap='Wistia', vmin=0.5, vmax=1)
        axs[i].imshow(masked_im[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].axis('off')

    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()

def create_mr_img(dwi_img, savefile, d, ext='png', dpi=250):
    dwi_img = np.rot90(dwi_img)
    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.4, wspace=0)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].axis('off')
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()

out_folder = 'D:/ctp_project_data/DWI_Training_Data_INSP/results/' + os.path.splitext(model_name)[0]
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

results = pd.DataFrame(columns=['id', 'dice', 'size'])
results['id'] = ['test_' + str(item).zfill(3) for item in range(1, len(images_res) + 1)]

for i in range(len(images_res)):
    print('Predicting for image {}.'.format(str(i + 1).zfill(2)))
    pred = (output[i].astype('bool') * 1).flatten()
    target = (targets_res[i].astype('bool')*1).flatten()
    size = np.sum(target)
    img = images_res[i]
    dice = f1_score(target, pred, pos_label=1, average='binary', zero_division=0)
    name = 'test_' + str(i + 1).zfill(3)
    # filename = os.path.join(out_folder,name + '_pred.png')
    # create_mrlesion_img(img, output[i], filename, define_dvalues(img), ext='png', dpi=250)
    # filename = os.path.join(out_folder, name + '_truth.png')
    # create_mrlesion_img(img, targets_res[i], filename, define_dvalues(img), ext='png', dpi=250)
    # filename = os.path.join(out_folder, name + '_dwi.png')
    # create_mr_img(img, filename, define_dvalues(img), ext='png', dpi=250)
    results.loc[results.id == name, 'size'] = size
    results.loc[results.id == name, 'dice'] = dice


results = results.join(ctp_df[~ctp_df.index.duplicated(keep='first')], how='left', on='id')

results.to_csv(os.path.join(out_folder, 'results.csv'), index=False)

    # make images from them?

