import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import glob
import matplotlib.pyplot as plt
from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01, normalize_01_clip99, AlbuSeg2d
from data_gen import SegmentationDataSet
from torch.utils import data
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
from torch.utils.data import DataLoader
import albumentations
import torch
from unet import UNet
from trainer import Trainer
from visual import plot_training
from loss import diceloss, diceloss1

'''
Place where we put it together
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()

#
# data_loc = 'D:/ctp_project_data/DWI_Training_Data/input_data_numpy/train'
#
# images = [os.path.join(data_loc, 'images', file)
#           for file in os.listdir(os.path.join(data_loc, 'images'))]
# masks = [os.path.join(data_loc, 'masks', file)
#          for file in os.listdir(os.path.join(data_loc, 'masks'))]
#
#
# training_dataset = SegmentationDataSet(inputs=images,
#                                        targets=masks,
#                                        transform=None)
#
# training_dataloader = data.DataLoader(dataset=training_dataset,
#                                       batch_size=2,
#                                       shuffle=True)
# x, y = next(iter(training_dataloader))
#
# print(f'x = shape: {x.shape}; type: {x.dtype}')
# print(f'x = min: {x.min()}; max: {x.max()}')
# print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')
#

# make transforms

if os.path.exists('D:'):
    root = pathlib.Path('D:/ctp_project_data/DWI_Training_Data_INSP/validation/')
elif os.path.exists('/media/'):
    root = '/media/mbcneuro/HDD1/DWI_Training_Data_INSP/validation/'


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# inputs_train = get_filenames_of_path(root/'train' / 'images')
# targets_train = get_filenames_of_path(root/'train' / 'masks')
#
# inputs_valid = get_filenames_of_path(root/'validation' / 'images')
# targets_valid = get_filenames_of_path(root/'validation' / 'masks')

inputs_train = glob.glob(os.path.join(root, 'train', 'images'))
targets_train = glob.glob(os.path.join(root, 'train', 'masks'))

inputs_valid = glob.glob(os.path.join(root, 'validation', 'images'))
targets_valid = glob.glob(os.path.join(root, 'validation', 'masks'))

# pre-transformations
pre_transforms = ComposeDouble([
    FunctionWrapperDouble(resize,
                          input=True,
                          target=False,
                          output_shape=(128, 128, 128)),
    FunctionWrapperDouble(resize,
                          input=False,
                          target=True,
                          output_shape=(128, 128, 128),
                          order=0,
                          anti_aliasing=False,
                          preserve_range=True),
])



# training transformations and augmentations
transforms_training = ComposeDouble([
    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.expand_dims, input=True, target=False, axis=0),
    #FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01_clip99, input=True, target=False)
])
transforms_validations = ComposeDouble([
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.expand_dims, input=True, target=False, axis=0),
    #FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01_clip99, input=True, target=False)
])

# random seed
random_seed = 42


# dataset training
dataset_train = SegmentationDataSet(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training,
                                    pre_transform=pre_transforms,
                                    use_cache=False)

# dataset validation
dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validations,
                                    pre_transform=pre_transforms,
                                    use_cache=False)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=2,
                                   shuffle=True)

x, y = next(iter(dataloader_training))
print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#
#     device = torch.device('cpu')
device =torch.device('cpu')
model = UNet(in_channels=1,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3).to(device)

# criterion
criterion = torch.nn.CrossEntropyLoss()
criterion = diceloss1(smooth=1.)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=10,
                  epoch=0)

# start training

training_losses, validation_losses, lr_rates = trainer.run_trainer()
fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10, 4))

from datetime import datetime
today = datetime.now()


# save the model
model_name =  'dwi_model_10epoch_diceloss_lr0.01_clip199.pt'
out_folder = 'D:/ctp_project_data/DWI_Training_Data_INSP/results/' + os.path.splitext(model_name)[0]
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

savefile = os.path.join(out_folder, 'training_validation_loss' + today.strftime("%m%d") + '.png')
plt.savefig(savefile,  facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=250, format='png')
