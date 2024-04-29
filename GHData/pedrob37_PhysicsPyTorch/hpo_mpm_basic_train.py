import numpy as np
import sys
import monai
import ponai
# sys.path.append('/nfs/home/pedro/portio')
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.model import nnUNet
import random
from model.metric import DiceLoss
import glob
import time
import nibabel as nib

import monai.visualize.img2tensorboard as img2tensorboard
sys.path.append('/nfs/home/pedro/RangerLARS/over9000')
# from over9000 import RangerLars

os.chdir('/nfs/home/pedro/PhysicsPyTorch')
import porchio
from early_stopping import pytorchtools
import runai.hpo

strategy = runai.hpo.Strategy.GridSearch
runai.hpo.init('/nfs/home/pedro/', 'stratification')


def soft_dice_score(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(2, len(y_pred.shape)))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return np.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch


# Function for proper handling of bools in argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def strUpper(v):
    if v in ('MPRAGE', 'mprage'):
        return 'MPRAGE'
    elif v in ('SPGR', 'spgr'):
        return 'SPGR'
    else:
        raise argparse.ArgumentTypeError('MPRAGE or SPGR expected.')


parser = argparse.ArgumentParser(description='Passing files + relevant directories')
parser.add_argument('--csv_label', type=str)
parser.add_argument('--images_dir', type=str)
parser.add_argument('--labels_dir', type=str)
parser.add_argument('--job_name', type=str)
parser.add_argument('--experiment_mode', type=str)
parser.add_argument("--physics_flag", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--patch_size", type=int, default=80)
parser.add_argument("--stratification_epsilon", type=float, default=0.05)
parser.add_argument("--uncertainty_flag", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--dropout_level", type=float, default=0.0)
parser.add_argument("--num_unc_passes", type=int, default=20)
parser.add_argument('--generation_type', type=strUpper, nargs='?', default='MPRAGE')
# parser.add_argument('--resolution', type=int)
arguments = parser.parse_args()


if True:
    config = runai.hpo.pick(
    grid=dict(stratification_epsilon=[0.1, 1, 10, 50, 100]),
    #        batch_size=[32, 64, 128],
    #        lr=[0.1, 0.01, 0.001],
    #        aug=[0.1, 0.2, 0.3],
    #        chns=[64, 128, 256],
    #        dropout=[0.1, 0.2, 0.3, 0.4, 0.5]),
    strategy=strategy)
else:
    config = dict( stratification_epsilon=arguments.stratification_epsilon)


def BespokeDataset(df, transform, patch_size, batch_seed, train=True, queue_length=4):
    loader = porchio.ImagesDataset
    sampler = porchio.data.UniformSampler(patch_size=patch_size, batch_seed=batch_seed)

    # These names are arbitrary
    MRI = 'mri'
    SEG = 'seg'
    PHYSICS = 'physics'

    subjects = []

    if train:
        for (image_path, label_path, subject_physics) in zip(df.Filename, df.Label_Filename, df.subject_physics):
            subject_dict = {
                MRI: porchio.ScalarImage(image_path),
                SEG: porchio.LabelMap(label_path),
                PHYSICS: subject_physics
            }
            subject = porchio.Subject(subject_dict)
            subjects.append(subject)
        this_dataset = loader(subjects, transform)
    else:
        for (image_path, subject_physics) in zip(df.Filename, df.subject_physics):
            subject_dict = {
                MRI: porchio.ScalarImage(image_path),
                PHYSICS: subject_physics
            }
            subject = porchio.Subject(subject_dict)
            subjects.append(subject)
        this_dataset = loader(subjects, transform)

    patches_dataset = porchio.Queue(
        subjects_dataset=this_dataset,
        max_length=queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        shuffle_subjects=False,
        shuffle_patches=False,
        num_workers=24,
    )

    return patches_dataset


# Not enough to shuffle batches, shuffle WITHIN batches!
# Take original csv, shuffle between subjects!
def reshuffle_csv(og_csv, batch_size):
    # Calculate some necessary variables
    batch_reshuffle_csv = pd.DataFrame({})
    num_images = len(og_csv)
    batch_numbers = list(np.array(range(num_images // batch_size)) * batch_size)
    num_unique_subjects = og_csv.subject_id.nunique()
    unique_subject_ids = og_csv.subject_id.unique()

    # First, re-order within subjects so batches don't always contain same combination of physics parameters
    for sub_ID in unique_subject_ids:
        batch_reshuffle_csv = batch_reshuffle_csv.append(og_csv[og_csv.subject_id == sub_ID].sample(frac=1).
                                                         reset_index(drop=True), ignore_index=True)

    # Set up empty lists for appending re-ordered entries
    new_subject_ids = []
    new_filenames = []
    new_label_filenames = []
    new_physics = []
    new_folds = []
    for batch in range(num_images // batch_size):
        # Randomly sample a batch ID
        batch_id = random.sample(batch_numbers, 1)[0]
        # Find those images/ labels/ params stipulated by the batch ID
        transferred_subject_ids = batch_reshuffle_csv.subject_id[batch_id:batch_id + batch_size]
        transferred_filenames = batch_reshuffle_csv.Filename[batch_id:batch_id + batch_size]
        transferred_label_filenames = batch_reshuffle_csv.Label_Filename[batch_id:batch_id + batch_size]
        transferred_physics = batch_reshuffle_csv.subject_physics[batch_id:batch_id + batch_size]
        transferred_folds = batch_reshuffle_csv.fold[batch_id:batch_id + batch_size]
        # Append these to respective lists
        new_subject_ids.extend(transferred_subject_ids)
        new_filenames.extend(transferred_filenames)
        new_label_filenames.extend(transferred_label_filenames)
        new_physics.extend(transferred_physics)
        new_folds.extend(transferred_folds)
        # Remove batch number used to reshuffle certain batches
        batch_numbers.remove(batch_id)

    altered_basic_csv = pd.DataFrame({
        'subject_id': new_subject_ids,
        'Filename': new_filenames,
        'subject_physics': new_physics,
        'fold': new_folds,
        'Label_Filename': new_label_filenames
    })
    return altered_basic_csv


def visualise_batch_patches(loader, bs, ps, comparisons=2):
    print('Calculating tester...')
    assert comparisons <= batch_size
    next_data = next(iter(loader))
    batch_samples = random.sample(list(range(bs)), comparisons)
    import matplotlib.pyplot as plt
    # Set up figure for ALL intra-batch comparisons
    f, axarr = plt.subplots(3, comparisons)
    for comparison in range(comparisons):
        # print(f'Label shape is {next_data["seg"]["data"].shape}')
        # print(f'Data shape is {next_data["mri"]["data"].shape}')
        example_batch_patch = np.squeeze(next_data['mri']['data'][batch_samples[comparison], ..., int(ps/2)])
        # For segmentation need to check that all classes (in 4D) have same patch that ALSO matches data
        example_batch_patch2 = np.squeeze(next_data['seg']['data'][batch_samples[comparison], 0, ..., int(ps/2)])
        example_batch_patch3 = np.squeeze(next_data['seg']['data'][batch_samples[comparison], 1, ..., int(ps/2)])
        axarr[0, comparison].imshow(example_batch_patch)
        axarr[0, comparison].axis('off')
        axarr[1, comparison].imshow(example_batch_patch2)
        axarr[1, comparison].axis('off')
        axarr[2, comparison].imshow(example_batch_patch3)
        axarr[2, comparison].axis('off')
    plt.show()


# Stratification specific functions
def feature_loss_func(volume1, volume2, tm):
    if tm == 'stratification':
        # if type(volume2) == np.ndarray:
        #     return np.mean((volume1 - volume2) ** 2)
        # else:
        feature_loss = torch.nn.MSELoss()
        flf = feature_loss(volume1, volume2)
        # old_flf = torch.mean((volume1 - volume2) ** 2).item()
        return flf
    elif tm == 'kld':
        kld = torch.nn.KLDivLoss()
        if type(volume2) == np.ndarray:
            raise TypeError
        else:
            return kld(volume1.detach().cpu(), volume2.detach().cpu())


def stratification_checker(input_volume):
    # Will only work for batch size 4 for now, but that comprises most experiments
    return int(torch.sum(input_volume[0, ...] + input_volume[3, ...] - input_volume[1, ...] - input_volume[2, ...]))


def dynamic_stratification_checker(input_volume):
    # Need to be able to calculate stratification regardless of batch size
    # Check batch size is at least 2
    current_batch_size = input_volume.shape[0]
    assert current_batch_size >= 2
    input_volume_shape = input_volume.shape
    zeros_volume = torch.zeros(size=input_volume_shape[1:]).cuda()
    if current_batch_size % 2 == 0:  # i.e.: Even
        for fhalf in range(current_batch_size//2):
            zeros_volume += input_volume[fhalf, ...]
        for lhalf in range(current_batch_size//2, current_batch_size):
            zeros_volume -= input_volume[lhalf, ...]
    else:
        multiplier = 1.0
        # Asymmetric, need to address in this manner
        zeros_volume += 0.5 * input_volume[0, ...]
        zeros_volume += 0.5 * input_volume[1, ...]
        zeros_volume -= input_volume[2, ...]
        # For remaining examples in batch, alternate: Should be even number remaining
        for rem in range(3, current_batch_size):
            zeros_volume += multiplier * input_volume[rem, ...]
            multiplier *= -1
    return torch.sum(zeros_volume)


def dynamic_calc_feature_loss(input_volume, tm):
    # Need to be able to calculate feature loss regardless of batch size
    import itertools
    current_batch_size = input_volume.shape[0]
    possible_combinations = list(itertools.combinations(range(current_batch_size), 2))
    overall_feature_loss = torch.empty(size=(len(possible_combinations),)).cuda()
    for iteration, comb in enumerate(possible_combinations):
        specific_feature_loss = feature_loss_func(input_volume[[comb[0]], ...], input_volume[[comb[1]], ...], tm=tm)
        # print(f'Specific feature loss shape is {specific_feature_loss.shape}')
        overall_feature_loss[iteration] = specific_feature_loss
    return torch.mean(overall_feature_loss)


def calc_feature_loss(input_volume, tm):
    feature_loss1 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[1, ...], tm=tm)
    feature_loss2 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[2, ...], tm=tm)
    feature_loss3 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[3, ...], tm=tm)
    feature_loss4 = feature_loss_func(
        volume1=input_volume[1, ...],
        volume2=input_volume[2, ...], tm=tm)
    feature_loss5 = feature_loss_func(
        volume1=input_volume[1, ...],
        volume2=input_volume[3, ...], tm=tm)
    feature_loss6 = feature_loss_func(
        volume1=input_volume[2, ...],
        volume2=input_volume[3, ...], tm=tm)

    total_feature_loss = torch.mean(torch.stack((feature_loss1,
                                                 feature_loss2,
                                                 feature_loss3,
                                                 feature_loss4,
                                                 feature_loss5,
                                                 feature_loss6)))
    return total_feature_loss


def normalise_image(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# torch.cuda.empty_cache()

# Writer will output to ./runs/ directory by default
log_dir = f'/nfs/home/pedro/PhysicsPyTorch/logger/logs/{arguments.job_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
SAVE_PATH = os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/logger/models/{arguments.job_name}')
FIG_PATH = os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/logger/Figures/{arguments.job_name}')
print(FIG_PATH)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)
SAVE = True
LOAD = True
patch_test = False
val_test = False

# Physics specific parameters
physics_flag = arguments.physics_flag
uncertainty_flag = arguments.uncertainty_flag
dropout_level = arguments.dropout_level
f'Physics is {physics_flag},  Uncertainty is {uncertainty_flag}'
physics_experiment_type = arguments.generation_type
print(f'The experiment type is {physics_experiment_type}')
physics_input_size = {'MPRAGE': 2,
                      'SPGR': 6}


def physics_preprocessing(physics_input, experiment_type):
    if experiment_type == 'MPRAGE':
        expo_physics = torch.exp(-physics_input)
        # print(f'The physics is {expo_physics.shape}, {physics_input.shape}')
        overall_physics = torch.stack((physics_input, expo_physics), dim=1)
    elif experiment_type == 'SPGR':
        TR_expo_params = torch.unsqueeze(torch.exp(-physics_input[:, 0]), dim=1)
        TE_expo_params = torch.unsqueeze(torch.exp(-physics_input[:, 1]), dim=1)
        FA_sin_params = torch.unsqueeze(torch.sin(physics_input[:, 2] * 3.14159265 / 180), dim=1)
        overall_physics = torch.cat((physics_input, torch.stack((TR_expo_params, TE_expo_params, FA_sin_params), dim=1).squeeze()), dim=1)
    return overall_physics


# Uncertainty: Pure noise
def corrected_paper_stochastic_loss(logits, sigma, labels, num_passes):
    total_loss = 0
    img_batch_size = logits.shape[0]
    logits_shape = list(logits.shape)
    # print(f'Logits sigma shape is {logits_shape} {sigma.shape}')
    # if not dropout_level != 0.0:
    # logits_shape.append(num_passes)
    # noise_array = torch.normal(mean=0.0, std=1.0, size=logits_shape, device=torch.device('cuda:0'))
    # vol_std = np.zeros((img_batch_size, num_passes))
    ax = torch.distributions.Normal(torch.tensor(0.0).to(device=torch.device("cuda:0")),
                                    torch.tensor(1.0).to(device=torch.device("cuda:0")))
    for fpass in range(num_passes):
        # if uncertainty_flag and dropout_level != 0.0:
        # print('You must have hetero. + DO on!')
        # noise_array = torch.normal(mean=0.0, std=1.0, size=logits_shape, device=torch.device('cuda:0'))
        # noise_array = torch.normal(mean=0.0, std=1.0, size=logits_shape, device=torch.device('cuda:0'))
        # stochastic_output = logits + sigma * noise_array  # * ax.sample(logits_shape)
        stochastic_output = logits + sigma * ax.sample(logits_shape)
        # else:
        #     stochastic_output = logits + sigma * noise_array[..., fpass]
        # temp_vol = torch.softmax(stochastic_output, dim=1)
        # temp_vol = temp_vol[:, 0, ...]
        # vol_std[:, fpass] = temp_vol.view(4, -1).sum(1).detach().cpu().numpy()
        # The dimension is the class dimension!!
        exponent_B = torch.log(torch.sum(torch.exp(stochastic_output), dim=1, keepdim=True))
        # print(f'Exponent B shape is {exponent_B.shape}')
        inner_logits = exponent_B - stochastic_output
        soft_inner_logits = labels * inner_logits
        total_loss += torch.exp(soft_inner_logits)  #.detach()  #.item()
        del exponent_B, inner_logits, soft_inner_logits #,noise_array
        # gc.collect()
        # See probability distributions: torch.distributions
        # Just expand and do at once: Casting
        # Call backward in loop
    mean_loss = total_loss / num_passes
    actual_loss = torch.mean(torch.log(mean_loss))
    # batch_std = np.std(vol_std, axis=1)
    batch_std = np.array([0] * img_batch_size)
    return actual_loss, batch_std


def corrected_paper_stochastic_loss_wip(logits, sigma, labels, num_passes):
    img_batch_size = logits.shape[0]
    logits_shape = list(logits[0, ...].shape)
    logits_shape.append(num_passes)
    # print(f'The logits shape is {logits_shape}')
    ax = torch.distributions.Normal(torch.tensor(0.0).to(device=torch.device("cuda:0")),
                                    torch.tensor(1.0).to(device=torch.device("cuda:0")))
    for unc_batch in range(img_batch_size):
        # noise_array = torch.normal(mean=0.0, std=1.0, size=logits_shape, device=torch.device('cuda:0'))
        print(torch.cuda.memory_allocated())
        expanded_logits = logits[unc_batch, ..., None].repeat((1, 1, 1, 1, num_passes))
        expanded_sigma = sigma[unc_batch, ..., None].repeat((1, 1, 1, 1, num_passes))
        expanded_labels = labels[unc_batch, ..., None].repeat((1, 1, 1, 1, num_passes))
        print(torch.cuda.memory_allocated())
        # print(f'The expanded logits shape is {expanded_logits.shape} {expanded_sigma.shape}')
        stochastic_output = expanded_logits + expanded_sigma * ax.sample(logits_shape)
        del expanded_sigma, expanded_logits
        exponent_B = torch.log(torch.sum(torch.exp(stochastic_output), dim=-2, keepdim=True))
        inner_logits = exponent_B - stochastic_output
        soft_inner_logits = expanded_labels * inner_logits
        # Sum across number of passes
        total_loss = torch.sum(torch.exp(soft_inner_logits), dim=-1)  #.detach()  #.item()
        del exponent_B, inner_logits, soft_inner_logits, expanded_labels
        mean_loss = total_loss / num_passes
        actual_loss = torch.mean(torch.log(mean_loss))
        # if unc_batch != (img_batch_size - 1):
        #     actual_loss.backward(retain_graph=True)
        # else:
        actual_loss.backward()
        del actual_loss
    optimizer.step()
    batch_std = np.array([0] * img_batch_size)
    return actual_loss, batch_std


def save_img(image, affine, filename):
    nifti_img = nib.Nifti1Image(image, affine)
    nib.save(nifti_img, filename)


# Check if SAVE_PATH is empty
file_list = os.listdir(path=SAVE_PATH)
num_files = len(file_list)


# Hyper-parameter loading: General parameters so doesn't matter which model file is loaded exactly
if LOAD and num_files > 0:
    model_files = glob.glob(os.path.join(SAVE_PATH, '*.pth'))
    for some_model_file in model_files:
        print(some_model_file)
    latest_model_file = max(model_files, key=os.path.getmtime)
    checkpoint = torch.load(latest_model_file, map_location=torch.device('cuda:0'))
    print(f'Loading {latest_model_file}!')
    loaded_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    running_iter = checkpoint['running_iter']
    EPOCHS = 61

    # Memory related variables
    batch_size = checkpoint['batch_size']
    queue_length = batch_size
    patch_size = checkpoint['patch_size']
    samples_per_volume = 1
else:
    running_iter = 0
    loaded_epoch = -1
    EPOCHS = 61

    # Memory related variables
    patch_size = arguments.patch_size
    batch_size = 4
    queue_length = batch_size
    samples_per_volume = 1

# Validation
validation_interval = 3
if uncertainty_flag:
    num_loss_passes = arguments.num_unc_passes
    print(f'The number of loss passes will be {num_loss_passes}')

# Stratification
training_modes = ['standard', 'stratification', 'kld', 'inference']
training_mode = arguments.experiment_mode
print(f'The training mode is {training_mode}')
stratification_epsilon = config['stratification_epsilon']
print(f'The stratification epsilon is {stratification_epsilon}')

# Some necessary variables
dataset_csv = arguments.csv_label
img_dir = arguments.images_dir  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
label_dir = arguments.labels_dir  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
print(img_dir)
print(label_dir)
val_batch_size = 4


# Read csv + add directory to filenames
df = pd.read_csv(dataset_csv)
df['Label_Filename'] = df['Filename']
df['Filename'] = img_dir + '/' + df['Filename'].astype(str)
df['Label_Filename'] = label_dir + '/' + 'Label_' + df['Label_Filename'].astype(str)
num_folds = 1 # df.fold.nunique()

# OOD csv
OOD_df = pd.read_csv('/nfs/home/pedro/PhysicsPyTorch/OOD_physics_csv_folds_limited.csv')
OOD_img_dir = '/nfs/project/pborges/SS_LowTD_MPRAGE_OOD_All_subjects/Restricted_30'
OOD_label_dir = '/nfs/project/pborges/Labels_LowTD_MPRAGE_OOD_All_subjects/Restricted_30'
OOD_df['Label_Filename'] = OOD_df['Filename'].str.replace('SS_', '')
OOD_df['Filename'] = OOD_img_dir + '/' + OOD_df['Filename'].astype(str)
# Label pointless, really, but load anyway
OOD_df['Label_Filename'] = OOD_label_dir + '/' + 'Label_' + OOD_df['Label_Filename'].astype(str)


# print(OOD_df)
# Image generation code is hiding under this comment
# On demand image generation
def mprage(T1, PD, TI, TD, tau, Gs=1):
    mprage_img = Gs * PD * (1 - 2 * np.exp(-TI / T1) / (1 + np.exp(-(TI + TD + tau) / T1)))
    return mprage_img


# Transforms
if arguments.generation_type == 'MPRAGE':
    training_transform = porchio.Compose([
        # porchio.RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        # porchio.RandomMotion(),
        # porchio.HistogramStandardization({MRI: landmarks}),
        porchio.RandomMPRAGE(TI=(0.6, 1.2), p=1),
        # porchio.RandomBiasField(coefficients=0.2),  # Bias field coeffs: Default 0.5 may be a bit too high!
        porchio.ZNormalization(masking_method=None),  # This is whitening
        # porchio.RandomNoise(std=(0, 0.1)),
        # porchio.ToCanonical(),
        # porchio.Resample((4, 4, 4)),
        # porchio.CropOrPad((48, 60, 48)),
        # porchio.RandomFlip(axes=(0,)),
        # porchio.OneOf({
        #     porchio.RandomAffine(): 0.8,
        #     porchio.RandomElasticDeformation(): 0.2,}),
    ])

    validation_transform = porchio.Compose([
        porchio.RandomMPRAGE(TI=(0.6, 1.2), p=1),
        porchio.ZNormalization(masking_method=None),
        # porchio.ToCanonical(),
        # porchio.Resample((4, 4, 4)),
        # porchio.CropOrPad((48, 60, 48)),
    ])

    inference_transform = porchio.Compose([
        porchio.ZNormalization(masking_method=None),
        # porchio.ToCanonical(),
        # porchio.Resample((4, 4, 4)),
        # porchio.CropOrPad((48, 60, 48)),
    ])
elif arguments.generation_type == 'SPGR':
    training_transform = porchio.Compose([
        # porchio.RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        # porchio.RandomMotion(),
        # porchio.HistogramStandardization({MRI: landmarks}),
        porchio.RandomSPGR(TR=(0.005, 2.0),
                           TE=(0.005, 0.1),
                           FA=(5.0, 90.0),
                           p=1),
        porchio.RandomBiasField(coefficients=0.2),  # Bias field coeffs: Default 0.5 may be a bit too high!
        porchio.ZNormalization(masking_method=None),  # This is whitening
        porchio.RandomNoise(std=(0, 0.1)),
        # porchio.ToCanonical(),
        # porchio.Resample((4, 4, 4)),
        # porchio.CropOrPad((48, 60, 48)),
        # porchio.RandomFlip(axes=(0,)),
        # porchio.OneOf({
        #     porchio.RandomAffine(): 0.8,
        #     porchio.RandomElasticDeformation(): 0.2,}),
    ])

    validation_transform = porchio.Compose([
        porchio.RandomSPGR(TR=(0.005, 2.0),
                           TE=(0.005, 0.1),
                           FA=(5.0, 90.0),
                           p=1),
        porchio.ZNormalization(masking_method=None),
        # porchio.ToCanonical(),
        # porchio.Resample((4, 4, 4)),
        # porchio.CropOrPad((48, 60, 48)),
    ])

    inference_transform = porchio.Compose([
        porchio.RandomSPGR(TR=(0.005, 2.0),
                           TE=(0.005, 0.1),
                           FA=(5.0, 90.0),
                           p=1),
        porchio.ZNormalization(masking_method=None),
        # porchio.ToCanonical(),
        # porchio.Resample((4, 4, 4)),
        # porchio.CropOrPad((48, 60, 48)),
    ])

# CUDA variables
use_cuda = torch.cuda.is_available()
print('Using cuda', use_cuda)

if use_cuda and torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs!')

stacked_cv = False
OOD_flag = True
if not stacked_cv:
    inf_fold = 5
    if not OOD_flag:
        inf_df = df[df.fold == inf_fold]
    else:
        inf_df = OOD_df[OOD_df.fold == inf_fold]
    inf_df.reset_index(drop=True, inplace=True)

# Loader for inference
inference_set = BespokeDataset(inf_df, inference_transform, patch_size=(181, 217, 181), batch_seed=1,
                               queue_length=batch_size)
inf_loader = DataLoader(inference_set, batch_size=1, shuffle=False)
# print(f'The inference set is {inference_set}')
# print(inference_set[0])

# For aggregation
overall_val_names = []
overall_val_metric = []
overall_gm_volumes = []
overall_gm_volumes2 = []

# If pretrained then initial model file will NOT match those created here: Therefore need to account for this
# Because won't be able to extract epoch and/ or fold from the name
if LOAD and num_files > 0:
    pretrained_checker = 'fold' in os.path.basename(latest_model_file)

# Find out fold and epoch
if LOAD and num_files > 0 and pretrained_checker:
    basename = os.path.basename(latest_model_file)
    if 'best' in basename:
        latest_epoch = int(os.path.splitext(basename)[0].split('_')[3])
        latest_fold = int(os.path.splitext(basename)[0].split('_')[5])
    else:
        latest_epoch = int(os.path.splitext(basename)[0].split('_')[2])
        latest_fold = int(os.path.splitext(basename)[0].split('_')[4])
    print(f'The latest epoch is {latest_epoch}, the loaded epoch is {loaded_epoch}')
    assert latest_epoch == loaded_epoch
else:
    latest_epoch = -1
    latest_fold = 0

print(f'\nStarted {training_mode}-ing!')
loop_switch = True
for fold in range(latest_fold, num_folds):
    while loop_switch:
        print('\nFOLD', fold)
        # Pre-loading sequence
        model = nnUNet(1, 2, physics_flag=physics_flag, physics_input=physics_input_size[physics_experiment_type],
                       physics_output=40, uncertainty_flag=uncertainty_flag, dropout_level=dropout_level)
        model = nn.DataParallel(model)
        # optimizer = RangerLars(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

        # Early Stopping
        early_stopping = pytorchtools.EarlyStopping(patience=9, verbose=True)

        # Running lists
        running_val_names = []
        running_val_metric = []
        running_gm_volumes = []
        running_gm_volumes2 = []

        # Specific fold writer
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold}'))

        if LOAD and num_files > 0 and training_mode != 'inference':
            # Get model file specific to fold
            loaded_model_file = f'model_epoch_{loaded_epoch}_fold_{fold}.pth'
            checkpoint = torch.load(os.path.join(SAVE_PATH, loaded_model_file), map_location=torch.device('cuda:0'))
            # Main model variables
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Get the validation entries from previous folds!
            running_val_names = checkpoint['running_val_names']
            running_val_metric = checkpoint['running_val_metric']
            running_gm_volumes = checkpoint['running_gm_volumes']
            try:
                running_gm_volumes2 = checkpoint['running_gm_volumes2']
            except:
                print('Missing running volumes2')
            overall_val_names = checkpoint['overall_val_names']
            overall_val_metric = checkpoint['overall_val_metric']
            overall_gm_volumes = checkpoint['overall_gm_volumes']
            try:
                overall_gm_volumes2 = checkpoint['overall_gm_volumes2']
            except:
                print('Missing overall volumes2')
            # Ensure that no more loading is done for future folds
            LOAD = False
        elif LOAD and num_files > 0 and training_mode == 'inference':
            # Get model file specific to fold
            # try:
            #     best_model_file = glob.glob(os.path.join(SAVE_PATH, f'best_model_epoch_*_fold_{fold}.pth'))
            # except:
            best_model_file = f'model_epoch_{loaded_epoch}_fold_{fold}.pth'
            print(f'Loading checkpoint for model: {best_model_file}')
            checkpoint = torch.load(os.path.join(SAVE_PATH, best_model_file), map_location=torch.device('cuda:0'))
            # Main model variables
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Get the validation entries from previous folds!
            running_val_names = checkpoint['running_val_names']
            running_val_metric = checkpoint['running_val_metric']
            running_gm_volumes = checkpoint['running_gm_volumes']
            try:
                running_gm_volumes2 = checkpoint['running_gm_volumes2']
            except:
                print('Missing running volumes2')
            overall_val_names = checkpoint['overall_val_names']
            overall_val_metric = checkpoint['overall_val_metric']
            overall_gm_volumes = checkpoint['overall_gm_volumes']
            try:
                overall_gm_volumes2 = checkpoint['overall_gm_volumes2']
            except:
                print('Missing overall volumes2')

        if stacked_cv:  # Pretty much never use this one
            # Train / Val/ Inf split
            val_fold = fold
            inf_fold = num_folds - fold - 1
            excluded_folds = [val_fold, inf_fold]
            train_df = df[~df.fold.isin(excluded_folds)]
            # if not OOD_flag:
            val_df = df[df.fold == val_fold]
            inf_df = df[df.fold == inf_fold]
            # else:
            #     val_df = OOD_df[OOD_df.fold == val_fold]
            #     inf_df = OOD_df[OOD_df.fold == inf_fold]
            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)
            inf_df.reset_index(drop=True, inplace=True)
        else:
            # Train / Val split
            val_fold = fold
            excluded_folds = [val_fold]
            train_df = df[~df.fold.isin(excluded_folds)]
            # if not OOD_flag:
            val_df = df[df.fold == val_fold]
            # else:
            #     val_df = OOD_df[OOD_df.fold == val_fold]
            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)

        print(f'The length of the training is {len(train_df)}')
        print(f'The length of the validation is {len(val_df)}')
        print(f'The length of the inference is {len(inf_df)}')

        model.cuda()
        print(f'\nStarted {training_mode}-ing!')
        for epoch in range(loaded_epoch, EPOCHS):
            print(f'Training Epoch: {epoch}')
            running_loss = 0.0
            model.train()
            train_acc = 0
            total_dice = 0
            new_seed = np.random.randint(10000)

            # Shuffle training and validation:
            new_train_df = reshuffle_csv(og_csv=train_df, batch_size=batch_size)
            new_val_df = reshuffle_csv(og_csv=val_df, batch_size=batch_size)

            # Val test
            if val_test:
                new_train_df = new_train_df[:20]

            # And generate new loaders
            patches_training_set = BespokeDataset(new_train_df, training_transform, patch_size, batch_seed=new_seed,
                                                  queue_length=batch_size)
            train_loader = DataLoader(patches_training_set, batch_size=batch_size, shuffle=False)
            patches_validation_set = BespokeDataset(new_val_df, validation_transform, patch_size, batch_seed=new_seed,
                                                    queue_length=val_batch_size)
            val_loader = DataLoader(patches_validation_set, batch_size=val_batch_size, shuffle=False)

            # Early stopping
            best_val_dice = 0.0
            best_counter = 0

            # Patch test
            if patch_test and epoch == 0 and fold == 0:
                visualise_batch_patches(loader=train_loader, bs=batch_size, ps=patch_size, comparisons=4)
            if training_mode != 'inference':
                for i, sample in enumerate(train_loader):
                    start = time.time()
                    images = sample['mri']['data'].cuda()
                    labels = sample['seg']['data'].cuda()
                    physics = sample['physics'].cuda().float().squeeze()
                    names = sample['mri']['path']
                    names = [os.path.basename(name) for name in names]

                    # print(f'The physics are {physics}')
                    # print(f'The physics shapes are {physics.shape}')
                    # print(f'The image shapes are {images.shape}')
                    # print(f'The names are {names}')
                    # Need to replace names to include physics (3 decimal points should suffice)
                    new_names = []
                    affine_array = np.array([[-1, 0, 0, 89],
                                             [0, 1, 0, -125],
                                             [0, 0, 1, -71],
                                             [0, 0, 0, 1]])
                    for k in range(4):
                        if physics_experiment_type == 'MPRAGE':
                            new_names.append(names[k].rsplit('.nii.gz')[0] + f'_TI_{physics[k]:.5f}' + '.nii.gz')
                        elif physics_experiment_type == 'SPGR':
                            new_names.append(names[k].rsplit('.nii.gz')[0] + f'_TR_{physics[k, 0]:.5f}'
                                             + f'_TE_{physics[k, 1]:.5f}'
                                             + f'_FA_{physics[k, 2]:.2f}'
                                             + '.nii.gz')
                        # save_img(images[k, ...].squeeze().detach().cpu().numpy(), affine_array,
                        #          os.path.join(FIG_PATH, os.path.basename(new_names[k])))
                        # print(f'The min and max of the images is {images[k, ...].squeeze().detach().cpu().numpy().min()},'
                        #       f'{images[k, ...].squeeze().detach().cpu().numpy().max()}')
                    names = new_names
                    # print(f'The new names are {names}')

                    # Zero grad optimizer
                    optimizer.zero_grad()
                    # print(images.shape, labels.shape, physics.shape)
                    # Pass images to the model
                    if not uncertainty_flag:
                        if physics_flag:
                            # Calculate physics extensions
                            processed_physics = physics_preprocessing(physics, physics_experiment_type)
                            # print(f'Processed physics shape is {processed_physics.shape}')
                            # print(processed_physics.shape, images.shape)
                            out, features_out = model(images, processed_physics)
                        else:
                            out, features_out = model(images)
                        # Loss
                        eps = 1e-10
                        loss_start = time.time()
                        data_loss = F.binary_cross_entropy_with_logits(out + eps, labels, reduction='mean')
                        loss_end = time.time()
                    else:
                        if physics_flag:
                            # Calculate physics extensions
                            processed_physics = physics_preprocessing(physics, physics_experiment_type)
                            # print(f'Processed physics shape is {processed_physics.shape}')
                            out, unc_out, features_out = model(images, processed_physics)
                        # print(f'Images shape is {images.shape}')
                        else:
                            out, unc_out, features_out = model(images)
                        loss_start = time.time()
                        data_loss, data_vol_std = corrected_paper_stochastic_loss(out, unc_out, labels,
                                                                                  num_passes=num_loss_passes)
                        loss_end = time.time()

                    if training_mode == 'standard':
                        loss = data_loss
                        total_feature_loss = 0.1 * dynamic_calc_feature_loss(
                            features_out, tm='stratification')  # NOTE: This needs to be the feature tensor!
                        writer.add_scalar('Loss/Feature_loss', total_feature_loss, running_iter)

                    elif training_mode == 'stratification' or training_mode == 'kld':
                        total_feature_loss = 0.1 * dynamic_calc_feature_loss(
                            features_out, tm=training_mode)  # NOTE: This needs to be the feature tensor!
                        # regulatory_ratio = data_loss / total_feature_loss

                        loss = data_loss + stratification_epsilon * total_feature_loss / (
                                1 + dynamic_stratification_checker(labels) * float(1e9)) ** 2
                        writer.add_scalar('Loss/Feature_loss', total_feature_loss, running_iter)

                    # Softmax to convert to probabilities
                    out = torch.softmax(out, dim=1)

                    # pGM = PairwiseMeasures(labels[:, 0, ...].detach().cpu().numpy(), out[:, 0, ...].detach().cpu().numpy())
                    # print(pGM.dice_score())
                    pGM_dice = soft_dice_score(labels.cpu().detach().numpy(), out.cpu().detach().numpy())
                    # print(pGM_dice)

                    # for param in model.parameters():
                    #     param.grad = None
                    # optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.detach().cpu().item()

                    # Name check: Shuffling sanity check
                    if i == 0:
                        print(f'The test names are: {names[0]}, {names[-2]}')

                    # Terminal logging
                    print(f"iter: {running_iter}, Loss: {loss.detach().item():.4f}, Dice: {pGM_dice:.3f}, "
                          f"strat: {stratification_checker(labels):.3f}"
                          f"                            ({loss_end - loss_start:.3f} s) ({(time.time() - start):.3f} s)")

                    # Writing to tensorboard
                    if running_iter % 50 == 0:
                        # Normalise images
                        images = normalise_image(images.cpu().detach().numpy())
                        out = normalise_image(out.cpu().detach().numpy())
                        labels = normalise_image(labels.cpu().detach().numpy())

                        writer.add_scalar('Loss/train', loss.detach().item(), running_iter)
                        img2tensorboard.add_animated_gif(writer=writer, image_tensor=images[0, ...],
                                                         tag=f'Visuals/Images_Fold_{fold}', max_out=patch_size // 2,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer, image_tensor=labels[0, 0, ...][None, ...],
                                                         tag=f'Visuals/Labels_Fold_{fold}', max_out=patch_size//2,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer, image_tensor=out[0, 0, ...][None, ...],
                                                         tag=f'Visuals/Output_Fold_{fold}', max_out=patch_size//2,
                                                         scale_factor=255, global_step=running_iter)
                        if uncertainty_flag:
                            unc_out = unc_out.cpu().detach().numpy()
                            unc_out = normalise_image(unc_out)
                            img2tensorboard.add_animated_gif(writer=writer, image_tensor=unc_out[0, 0, ...][None, ...],
                                                             tag=f'Validation/Unc_Output_Fold_{fold}', max_out=patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)

                    running_iter += 1
                    del sample, images, labels, physics, names, out, features_out
                    if uncertainty_flag:
                        del unc_out
                    import gc
                    gc.collect()

                print("Epoch: {}, Loss: {},\n Train Dice: Not implemented".format(epoch, running_loss))

                print('Validation step')
                model.eval()
                val_metric = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=False, softmax=True)
                val_running_loss = 0
                # correct = 0
                val_counter = 0
                names_collector = []
                metric_collector = []
                metric_collector2 = []
                gm_volumes_collector = []
                gm_volumes_collector2 = []
                CoV_collector = []
                CoV_collector2 = []

                val_start = time.time()
                if epoch % validation_interval == 0:
                    with torch.no_grad():
                        for val_sample in val_loader:
                            val_images = val_sample['mri']['data'].squeeze().cuda()
                            val_names = val_sample['mri']['path']
                            # Readjust dimensions to match expected shape for network
                            # if len(val_images.shape) == 3:
                            #     val_images = torch.unsqueeze(torch.unsqueeze(val_images, 0), 0)
                            # elif len(val_images.shape) == 4:
                            #     val_images = torch.unsqueeze(val_images, 0)
                            val_labels = val_sample['seg']['data'].squeeze().cuda()
                            # print(f'val_images shape is {val_images.shape}')
                            # print(f'val_labels shape is {val_labels.shape}')
                            # Readjust dimensions to match expected shape
                            if len(val_labels.shape) == 4:
                                val_labels = torch.unsqueeze(val_labels, 1)
                            if len(val_images.shape) == 4:
                                val_images = torch.unsqueeze(val_images, 1)
                            val_physics = val_sample['physics'].squeeze().cuda().float()
                            val_names = [os.path.basename(val_name) for val_name in val_names]

                            new_names = []
                            affine_array = np.array([[-1, 0, 0, 89],
                                                     [0, 1, 0, -125],
                                                     [0, 0, 1, -71],
                                                     [0, 0, 0, 1]])
                            for k in range(4):
                                if physics_experiment_type == 'MPRAGE':
                                    new_names.append('val_' + val_names[k].rsplit('.nii.gz')[0]
                                                     + f'_TI_{val_physics[k]:.5f}'
                                                     + '.nii.gz')
                                elif physics_experiment_type == 'SPGR':
                                    new_names.append('val_' + val_names[k].rsplit('.nii.gz')[0]
                                                     + f'_TR_{val_physics[k, 0]:.5f}'
                                                     + f'_TE_{val_physics[k, 1]:.5f}'
                                                     + f'_FA_{val_physics[k, 2]:.2f}'
                                                     + '.nii.gz')
                                save_img(val_images[k, ...].squeeze().detach().cpu().numpy(), affine_array,
                                         os.path.join(FIG_PATH, os.path.basename(new_names[k])))
                            val_names = new_names

                            # Small name check
                            # print(f'Val names are {val_names}')

                            # Pass images to the model
                            if not uncertainty_flag:
                                if physics_flag:
                                    # Calculate physics extensions
                                    val_processed_physics = physics_preprocessing(val_physics, physics_experiment_type)
                                    out, features_out = model(val_images, val_processed_physics)
                                else:
                                    out, features_out = model(val_images)
                                val_data_loss = F.binary_cross_entropy_with_logits(out, val_labels, reduction="mean")
                            else:
                                if physics_flag:
                                    # Calculate physics extensions
                                    val_processed_physics = physics_preprocessing(val_physics, physics_experiment_type)
                                    # print(f'Processed physics shape is {processed_physics.shape}')
                                    out, unc_out, features_out = model(val_images, val_processed_physics)
                                else:
                                    out, unc_out, features_out = model(val_images)
                                val_data_loss, val_data_vol_std = corrected_paper_stochastic_loss(out, unc_out, val_labels,
                                                                                                  num_passes=num_loss_passes)

                            # Loss depends on training mode
                            if training_mode == 'standard':
                                val_loss = val_data_loss
                            elif training_mode == 'stratification' or training_mode == 'kld':
                                val_total_feature_loss = 0.1 * dynamic_calc_feature_loss(
                                    features_out, tm=training_mode)  # NOTE: This needs to be the feature tensor!
                                # regulatory_ratio = val_data_loss / val_total_feature_loss
                                val_loss = val_data_loss + stratification_epsilon * val_total_feature_loss / (
                                            1 + dynamic_stratification_checker(val_labels) * float(1e9)) ** 2

                            # print(f"out val shape is {out.shape}")  # Checking for batch dimension inclusion or not
                            out = torch.softmax(out, dim=1)
                            gm_out = out[:, 0, ...]

                            val_running_loss += val_loss.detach().item()

                            # Metric calculation
                            # print(pGM_dice)
                            # dice_performance = val_metric.forward(out, val_labels)
                            gm_volume = gm_out.view(4, -1).sum(1)
                            names_collector += val_names
                            gm_volumes_collector += gm_volume

                            # Calculate CoVs
                            gm_volume_np = gm_volume.cpu().detach().numpy()
                            val_CoV = np.std(gm_volume_np) / np.mean(gm_volume_np)
                            for i in range(val_batch_size):
                                pGM_dice = soft_dice_score(val_labels[i, ...].cpu().detach().numpy(), out[i, ...].cpu().detach().numpy())
                                metric_collector += [pGM_dice.tolist()]
                                CoV_collector.append(val_CoV)
                            # writer.add_scalar('Loss/Val_Feature_loss', val_total_feature_loss, running_iter)

                            # Convert to numpy arrays
                            val_images = val_images.cpu().detach().numpy()
                            val_labels = val_labels.cpu().detach().numpy()
                            val_images = normalise_image(val_images)
                            out = out.cpu().detach().numpy()
                            out = normalise_image(out)

                            val_counter += val_batch_size  #Should probably be one to properly match training

                            # Cleaning up
                            # del val_sample, val_images, val_labels, val_physics, val_names

                    print(f'This validation step took {time.time() - val_start} s')
                    # Write to tensorboard
                    writer.add_scalar('Loss/val', val_running_loss / val_counter, running_iter)
                    writer.add_scalar('Loss/dice_val', np.mean(metric_collector), running_iter)
                    writer.add_scalar('Loss/CoV', np.mean(CoV_collector), running_iter)
                    writer.add_scalar('Loss/CoV2', np.mean(CoV_collector2), running_iter)
                    img2tensorboard.add_animated_gif(writer=writer, image_tensor=val_images[0, ...],
                                                     tag=f'Validation/Images_Fold_{fold}', max_out=patch_size // 4,
                                                     scale_factor=255, global_step=running_iter)
                    img2tensorboard.add_animated_gif(writer=writer, image_tensor=val_labels[0, 0, ...][None, ...],
                                                     tag=f'Validation/Labels_Fold_{fold}', max_out=patch_size // 4,
                                                     scale_factor=255, global_step=running_iter)
                    img2tensorboard.add_animated_gif(writer=writer, image_tensor=out[0, 0, ...][None, ...],
                                                     tag=f'Validation/Output_Fold_{fold}', max_out=patch_size // 4,
                                                     scale_factor=255, global_step=running_iter)
                    if uncertainty_flag:
                        unc_out = unc_out.cpu().detach().numpy()
                        unc_out = normalise_image(unc_out)
                        img2tensorboard.add_animated_gif(writer=writer, image_tensor=unc_out[0, 0, ...][None, ...],
                                                         tag=f'Validation/Unc_Output_Fold_{fold}', max_out=patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)

                    # Check if current val dice is better than previous best
                    # true_dice = np.mean(metric_collector)
                    # true_val = val_running_loss / val_counter  # alternative
                    # if true_dice > best_val_dice:
                    #     best_val_dice = true_dice
                    #     append_string = 'not_best'
                    #     best_counter = 0
                    # else:
                    #     append_string = 'nb'
                    #     best_counter += 1

                    # Aggregation
                    running_val_metric.append(np.mean(metric_collector))
                    running_val_names.append(names_collector)
                    running_gm_volumes.append(gm_volumes_collector)
                    running_gm_volumes2.append(gm_volumes_collector2)

                    # # Save model
                    # if SAVE and append_string == 'best':
                    #     MODEL_PATH = os.path.join(SAVE_PATH, f'model_epoch_{epoch}_fold_{fold}.pth')
                    #     print(MODEL_PATH)
                    #     torch.save({'model_state_dict': model.state_dict(),
                    #                 'optimizer_state_dict': optimizer.state_dict(),
                    #                 'epoch': epoch,
                    #                 'loss': loss,
                    #                 'running_iter': running_iter,
                    #                 'batch_size': batch_size,
                    #                 'patch_size': patch_size,
                    #                 'running_val_names': running_val_names,
                    #                 'running_val_metric': running_val_metric,
                    #                 'overall_val_names': overall_val_names,
                    #                 'overall_val_metric': overall_val_metric}, MODEL_PATH)

                    # Saving in-training csv
                    # print(f'The problematic names seem to be {running_val_names[-1]}')
                    current_subject_ids = [int(vn.rsplit('.nii.gz')[0].split('_')[4]) for vn in running_val_names[-1]]
                    print(len(running_val_names[-1]),  # running_val_names,
                          len(current_subject_ids),  # current_subject_ids,
                          len(metric_collector),  # metric_collector,
                          len(CoV_collector),  # CoV_collector,
                          len(running_gm_volumes[-1]),
                          len(running_gm_volumes2[-1]))
                    sub = pd.DataFrame({"Filename": running_val_names[-1],  # Done
                                        "subject_id": current_subject_ids,  # Done
                                        "Dice": metric_collector,
                                        "CoV": CoV_collector,
                                        "GM_volumes": running_gm_volumes[-1],
                                        'GM_volumes2': running_gm_volumes[-1]})

                    sub.to_csv(os.path.join(SAVE_PATH, f'val_epoch_{epoch}_fold_{fold}.csv'), index=False)

                    # Saving the model
                    # Save model
                    if SAVE:
                        MODEL_PATH = os.path.join(SAVE_PATH, f'model_epoch_{epoch}_fold_{fold}.pth')
                        print(MODEL_PATH)
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'epoch': epoch,
                                    'loss': loss,
                                    'running_iter': running_iter,
                                    'batch_size': batch_size,
                                    'patch_size': patch_size,
                                    'running_val_names': running_val_names,
                                    'running_val_metric': running_val_metric,
                                    'running_gm_volumes': running_gm_volumes,
                                    'running_gm_volumes2': running_gm_volumes2,
                                    'overall_gm_volumes': overall_gm_volumes,
                                    'overall_gm_volumes2': overall_gm_volumes2,
                                    'overall_val_names': overall_val_names,
                                    'overall_val_metric': overall_val_metric}, MODEL_PATH)

                    # Early stopping
                    early_stopping((np.mean(metric_collector)+val_batch_size*np.mean(CoV_collector)), model)

                    if early_stopping.early_stop or epoch == 60:
                        # Set overalls to best epoch
                        best_epoch = int(np.argmax(running_val_metric))
                        print(f'The best epoch is Epoch {best_epoch}')
                        overall_val_metric.append(running_val_metric[best_epoch])
                        overall_val_names.extend(running_val_names[best_epoch])
                        overall_gm_volumes.extend(running_gm_volumes[best_epoch])
                        overall_gm_volumes2.extend(running_gm_volumes2[best_epoch])

                        f = open(os.path.join(SAVE_PATH, f"Best_epoch_{best_epoch}.txt", "w"))
                        f.write(" Created file")
                        f.close()
                        print('Early stopping!')
                        break
                    del val_sample, val_images, val_labels, val_physics, val_names, out, features_out
                    if uncertainty_flag:
                        del unc_out

                else:
                    # Aggregation: Fill with some values so can actually match to best epoch
                    running_val_metric.append(-1e10)
                    running_val_names.append(-1e10)
                    running_gm_volumes.append(-1e10)
                    running_gm_volumes2.append(-1e10)
            elif training_mode == 'inference':
                model.eval()
                with torch.no_grad():
                    from ponai.inferers import sliding_window_inference
                    for inf_sample in inference_set:
                        # Variables from sampler
                        if not uncertainty_flag:
                            inf_names = inf_sample['mri']['path']
                            input_tensor = inf_sample['mri']['data'].squeeze().cuda()
                            inf_physics = torch.FloatTensor([inf_sample['physics']]).squeeze().cuda().float()
                            if len(inf_physics.shape) <= 4:
                                inf_physics = torch.unsqueeze(inf_physics, 0)
                            # Manually do Z normalisation
                            ip_mean = input_tensor.mean()
                            ip_std = input_tensor.std()
                            input_tensor -= ip_mean
                            input_tensor /= ip_std
                            overlap = 0.3
                            if len(input_tensor.shape) <= 4:
                                input_tensor = torch.unsqueeze(input_tensor, 0)
                                if len(input_tensor.shape) == 4:
                                    input_tensor = torch.unsqueeze(input_tensor, 0)
                            if physics_flag:
                                # print(f'ponai: {input_tensor.shape}, {inf_physics.shape}')
                                inf_processed_physics = physics_preprocessing(inf_physics, physics_experiment_type)
                                val_outputs = sliding_window_inference(
                                    (input_tensor, inf_processed_physics), 160, 1, model, overlap=overlap, mode='gaussian')
                            else:
                                val_outputs = sliding_window_inference(
                                    input_tensor, 160, 1, model, overlap=overlap, mode='gaussian')
                            out = torch.squeeze(torch.softmax(val_outputs, dim=1))
                            # print(f'Out shape is {out.shape}')
                            gm_output_tensor = out[0, ...]
                            save_img(np.squeeze(gm_output_tensor.cpu().numpy()), np.eye(4),
                                     os.path.join(FIG_PATH, f'monai_grid_fold_{fold}_' + os.path.basename(inf_names)))
                        else:
                            inf_names = inf_sample['mri']['path']
                            input_tensor = inf_sample['mri']['data'].squeeze().cuda()
                            inf_physics = torch.FloatTensor([inf_sample['physics']]).squeeze().cuda().float()
                            if len(inf_physics.shape) <= 4:
                                inf_physics = torch.unsqueeze(inf_physics, 0)
                            # Manually do Z normalisation
                            ip_mean = input_tensor.mean()
                            ip_std = input_tensor.std()
                            input_tensor -= ip_mean
                            input_tensor /= ip_std
                            overlap = 0.3
                            if len(input_tensor.shape) <= 4:
                                input_tensor = torch.unsqueeze(input_tensor, 0)
                                if len(input_tensor.shape) == 4:
                                    input_tensor = torch.unsqueeze(input_tensor, 0)
                            if physics_flag:
                                print(f'ponai: {input_tensor.shape}, {inf_physics.shape}')
                                inf_processed_physics = physics_preprocessing(inf_physics, physics_experiment_type)
                                val_outputs, unc_val_outputs = sliding_window_inference(
                                    (input_tensor, inf_processed_physics), 160, 1, model, overlap=overlap, mode='gaussian',
                                    uncertainty_flag=uncertainty_flag)
                            else:
                                val_outputs, unc_val_outputs = sliding_window_inference(
                                    input_tensor, 160, 1, model, overlap=overlap, mode='gaussian',
                                    uncertainty_flag=uncertainty_flag)
                            out = torch.squeeze(val_outputs)
                            gm_output_tensor = out[0, ...]
                            unc_output_tensor = unc_val_outputs[0, 0, ...]
                            save_img(np.squeeze(gm_output_tensor.cpu().numpy()), np.eye(4),
                                     os.path.join(FIG_PATH, f'monai_grid_fold_{fold}_' + os.path.basename(inf_names)))
                            save_img(np.squeeze(unc_output_tensor.cpu().numpy()), np.eye(4),
                                     os.path.join(FIG_PATH, f'unc_monai_grid_fold_{fold}_' + os.path.basename(inf_names)))
                        print(f'Running inference on: {os.path.basename(inf_names)}')
                    break

        # Now that this fold's training has ended, want starting points of next fold to reset
        latest_epoch = -1
        latest_fold = 0
        running_iter = 0
        loaded_epoch = 0

        # Model deletion and cache clearing to prevent polluting
        del model
        torch.cuda.empty_cache()

        if training_mode == 'inference':
            loop_switch = False

## Totals: What to collect after training has finished
# Dice for all validation? Volumes and COVs?

overall_val_metric = np.array(overall_val_metric)
overall_gm_volumes = np.array(overall_gm_volumes)
overall_gm_volumes2 = np.array(overall_gm_volumes2)
print(overall_val_names)
# Problem is likely in here!
overall_subject_ids = [int(vn[0].rsplit('.nii.gz')[0].split('_')[3]) for vn in overall_val_names]

# Folds analysis
print('Names', len(overall_val_names), 'Dice', len(overall_val_metric), 'GM volumes', len(overall_gm_volumes))

# Folds Dice
print('Overall Dice:', np.mean(overall_val_metric), 'std:', np.std(overall_val_metric))

sub = pd.DataFrame({"Filename": overall_val_names,
                    "subject_id": overall_subject_ids,
                    "Dice": overall_val_metric.tolist(),
                    "GM_volumes": overall_gm_volumes,
                    "GM_volumes2": overall_gm_volumes2})

sub.to_csv(os.path.join(SAVE_PATH, 'dice_gm_volumes.csv'), index=False)

# CoVs
subject_CoVs = []
subject_dice = []
for ID in range(sub.subject_id.nunique()):
    # CoV, see: https://en.wikipedia.org/wiki/Coefficient_of_variation
    subject_CoV = np.std(sub[sub.subject_id == ID].GM_volumes) / np.mean(sub[sub.subject_id == ID].GM_volumes)
    subject_CoVs.append(subject_CoV)
    subject_dice.append(np.mean(sub[sub.subject_id == ID].Dice))
    print(f'The CoV for subject {ID} is {subject_CoV}')
cov_sub = pd.DataFrame({"subject_id": list(range(sub.subject_id.nunique())),
                        "subject_dice": subject_dice,
                        "subject_CoVs": subject_CoVs})
print(f"The mean and std of all subject CoVs is: {np.mean(subject_CoVs)}, {np.std(subject_CoVs)}")

# HPO logging
runai.hpo.report(epoch=best_epoch, metrics={'val_dice_std': np.std(overall_val_metric),
                                            'val_dice': np.mean(overall_val_metric),
                                            'val_cov': np.mean(subject_CoVs),
                                            'val_cov_dice': np.mean(overall_val_metric)+4*np.mean(subject_CoVs)})
print('Finished!')
