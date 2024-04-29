import numpy as np
import monai
import porchio
from porchio import Queue
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
from ignite.handlers import EarlyStopping
from model.metric import DiceLoss
import glob
import time

import monai.visualize.img2tensorboard as img2tensorboard
import sys
sys.path.append('/home/pedro/over9000')
from over9000 import RangerLars


class BadDataset:
    def __init__(self, df, transform):
        self.df = df
        self.loader = porchio.ImagesDataset
        self.transform = transform
        self.sampler = porchio.data.UniformSampler(patch_size=80)

    def __getitem__(self, index):
        # These names are arbitrary
        MRI = 'mri'
        SEG = 'seg'
        PHYSICS = 'physics'

        subjects = []
        for (image_path, label_path, subject_physics) in zip(self.df.Filename, self.df.Label_Filename,
                                                             self.df.subject_physics):
            subject_dict = {
                MRI: porchio.ScalarImage(image_path),
                SEG: porchio.LabelMap(label_path),
                PHYSICS: subject_physics
            }
            subject = porchio.Subject(subject_dict)
            subjects.append(subject)
        this_dataset = self.loader(subjects, self.transform)

        patches_dataset = porchio.Queue(
            subjects_dataset=this_dataset,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=porchio.sampler.UniformSampler(patch_size),
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        return patches_dataset

    def __len__(self):
        return self.df.shape[0]


def BespokeDataset(df, transform, patch_size, batch_seed):
    loader = porchio.ImagesDataset
    sampler = porchio.data.UniformSampler(patch_size=patch_size, batch_seed=batch_seed)

    # These names are arbitrary
    MRI = 'mri'
    SEG = 'seg'
    PHYSICS = 'physics'

    subjects = []
    for (image_path, label_path, subject_physics) in zip(df.Filename, df.Label_Filename, df.subject_physics):
        subject_dict = {
            MRI: porchio.ScalarImage(image_path),
            SEG: porchio.LabelMap(label_path),
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
def feature_loss_func(volume1, volume2):
    if type(volume2) == np.ndarray:
        return np.mean((volume1 - volume2) ** 2)
    else:
        return torch.mean((volume1 - volume2) ** 2).item()


def stratification_checker(input_volume):
    # Will only work for batch size 4 for now, but that comprises most experiments
    return int(torch.sum(input_volume[0, ...] + input_volume[3, ...] - input_volume[1, ...] - input_volume[2, ...]))


def calc_feature_loss(input_volume):
    feature_loss1 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[1, ...])
    feature_loss2 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[2, ...])
    feature_loss3 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[3, ...])
    feature_loss4 = feature_loss_func(
        volume1=input_volume[1, ...],
        volume2=input_volume[2, ...])
    feature_loss5 = feature_loss_func(
        volume1=input_volume[1, ...],
        volume2=input_volume[3, ...])
    feature_loss6 = feature_loss_func(
        volume1=input_volume[2, ...],
        volume2=input_volume[3, ...])

    total_feature_loss = np.mean([feature_loss1,
                                 feature_loss2,
                                 feature_loss3,
                                 feature_loss4,
                                 feature_loss5,
                                 feature_loss6])
    return total_feature_loss


def normalise_image(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.cuda.empty_cache()

# Writer will output to ./runs/ directory by default
log_dir = f'/home/pedro/PhysicsPyTorch/logger/preliminary_tests_physics'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
SAVE_PATH = os.path.join(f'/home/pedro/PhysicsPyTorch/logger/preliminary_tests/models')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
SAVE = True
LOAD = True
patch_test = False
val_test = False

# Physics specific parameters
physics_flag = True
physics_experiment_type = 'MPRAGE'
physics_input_size = {'MPRAGE': 2,
                      'SPGR': 6}


def physics_preprocessing(physics_input, experiment_type):
    if experiment_type == 'MPRAGE':
        expo_physics = torch.exp(-physics_input)
        overall_physics = torch.stack((physics, expo_physics), dim=1)
    elif experiment_type == 'SPGR':
        TR_expo_params = torch.unsqueeze(torch.exp(-physics_input[:, 0]), dim=1)
        TE_expo_params = torch.unsqueeze(torch.exp(-physics_input[:, 1]), dim=1)
        FA_sin_params = torch.unsqueeze(torch.sin(physics_input[:, 2] * 3.14159265 / 180), dim=1)
        overall_physics = torch.stack((physics, TR_expo_params, TE_expo_params, FA_sin_params), dim=1)
    return overall_physics


# Check if SAVE_PATH is empty
file_list = os.listdir(path=SAVE_PATH)
num_files = len(file_list)


# Hyper-parameter loading: General parameters so doesn't matter which model file is loaded exactly
if LOAD and num_files > 0:
    model_files = glob.glob(os.path.join(SAVE_PATH, '*.pth'))
    latest_model_file = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest_model_file, map_location=torch.device('cuda:0'))
    print(f'Loading {latest_model_file}!')
    loaded_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    running_iter = checkpoint['running_iter']
    EPOCHS = 100

    # Memory related variables
    batch_size = checkpoint['batch_size']
    queue_length = batch_size
    patch_size = checkpoint['patch_size']
    samples_per_volume = 1
else:
    running_iter = 0
    loaded_epoch = -1
    EPOCHS = 100

    # Memory related variables
    patch_size = 16
    batch_size = 4
    queue_length = batch_size
    samples_per_volume = 1

# Stratification
training_modes = ['standard', 'stratification']
training_mode = 'stratification'
stratification_epsilon = 0.05

# Some necessary variables
dataset_csv = '/home/pedro/PhysicsPyTorch/local_physics_csv.csv'
# img_dir = '/data/MPRAGE_subjects_121T/Train_121T'  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
# label_dir = '/data/Segmentation_MPRAGE_121T/All_labels'  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
img_dir = '/data/Resampled_Data/Images/SS_GM_Images'  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
label_dir = '/data/Resampled_Data/Labels/GM_Labels'  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
print(img_dir)
print(label_dir)
val_batch_size = 4

# Read csv + add directory to filenames
df = pd.read_csv(dataset_csv)
df['Label_Filename'] = df['Filename']
df['Filename'] = img_dir + '/' + df['Filename'].astype(str)
df['Label_Filename'] = label_dir + '/' + 'Label_' + df['Label_Filename'].astype(str)
num_folds = df.fold.nunique()

# Transforms
training_transform = porchio.Compose([
    # porchio.RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
    # porchio.RandomMotion(),
    # porchio.HistogramStandardization({MRI: landmarks}),
    porchio.RandomBiasField(),
    porchio.ZNormalization(masking_method=None),
    porchio.RandomNoise(),
    # porchio.ToCanonical(),
    # porchio.Resample((4, 4, 4)),
    # porchio.CropOrPad((48, 60, 48)),
    # porchio.RandomFlip(axes=(0,)),
    # porchio.OneOf({
    #     porchio.RandomAffine(): 0.8,
    #     porchio.RandomElasticDeformation(): 0.2,}),
])

validation_transform = porchio.Compose([
    # porchio.HistogramStandardization({MRI: landmarks}),
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
if not stacked_cv:
    inf_fold = 5
    inf_df = df[df.fold == inf_fold]
    inf_df.reset_index(drop=True, inplace=True)

# For aggregation
overall_val_names = []
overall_val_metric = []
overall_gm_volumes = []


print(f'\nStarted {training_mode}-ing!')
for fold in range(num_folds):
    print('\nFOLD', fold)
    # Pre-loading sequence
    model = nnUNet(1, 2, physics_flag=physics_flag, physics_input=physics_input_size[physics_experiment_type],
                   physics_output=40)
    model = nn.DataParallel(model)
    optimizer = RangerLars(model.parameters())

    # Running lists
    running_val_names = []
    running_val_metric = []
    running_gm_volumes = []

    # Specific fold writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold}'))

    if LOAD and num_files > 0:
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
        overall_val_names = checkpoint['overall_val_names']
        overall_val_metric = checkpoint['overall_val_metric']
        overall_gm_volumes = checkpoint['overall_gm_volumes']
        # Ensure that no more loading is done for future folds
        LOAD = False

    if stacked_cv:  # Pretty much never use this one
        # Train / Val/ Inf split
        val_fold = fold
        inf_fold = num_folds - fold - 1
        excluded_folds = [val_fold, inf_fold]
        train_df = df[~df.fold.isin(excluded_folds)]
        val_df = df[df.fold == val_fold]
        inf_df = df[df.fold == inf_fold]
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        inf_df.reset_index(drop=True, inplace=True)
    else:
        # Train / Val split
        val_fold = fold
        excluded_folds = [val_fold]
        train_df = df[~df.fold.isin(excluded_folds)]
        val_df = df[df.fold == val_fold]
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

    print(f'The length of the training is {len(train_df)}')
    print(f'The length of the validation is {len(val_df)}')
    print(f'The length of the validation is {len(inf_df)}')

    model.cuda()
    print(f'\nStarted {training_mode}-ing!')
    for epoch in range(0, EPOCHS):
        print('Training Epoch')
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
        patches_training_set = BespokeDataset(new_train_df, training_transform, patch_size, batch_seed=new_seed)
        train_loader = DataLoader(patches_training_set, batch_size=batch_size, shuffle=False)
        patches_validation_set = BespokeDataset(new_val_df, validation_transform, patch_size, batch_seed=new_seed)
        val_loader = DataLoader(patches_validation_set, batch_size=val_batch_size)

        # Early stopping
        best_val_dice = 0.0
        best_counter = 0

        # Patch test
        if patch_test and epoch == 0 and fold == 0:
            visualise_batch_patches(loader=train_loader, bs=batch_size, ps=patch_size, comparisons=4)
        for i, sample in enumerate(train_loader):
            images = sample['mri']['data'].cuda()
            labels = sample['seg']['data'].cuda()
            physics = sample['physics'].cuda().float()
            names = sample['mri']['path']
            names = [os.path.basename(name) for name in names]

            # Pass images to the model
            start = time.time()
            if physics_flag:
                # Calculate physics extensions
                processed_physics = physics_preprocessing(physics, physics_experiment_type)
                # print(f'Processed physics shape is {processed_physics.shape}')
                out, features_out = model(images, processed_physics)
            # print(f'Images shape is {images.shape}')
            else:
                out, features_out = model(images)

            # Need loss here
            eps = 1e-10
            data_loss = F.binary_cross_entropy_with_logits(out+eps, labels, reduction='mean')

            # Some checks for the labels
            # print(f'{i}: Labels shape is {labels.shape}, the names are {names}')

            if training_mode == 'standard':
                loss = data_loss
                print(f"iter: {running_iter}, Loss: {loss.item():.4f},"
                      f"                                           ({(time.time() - start):.3f}s)")
            elif training_mode == 'stratification':
                total_feature_loss = 0.1 * calc_feature_loss(features_out)  # NOTE: This needs to be the feature tensor!
                regulatory_ratio = data_loss / total_feature_loss
                # print(f'The label directories are {sample}')
                print(f'The stratification check value is {stratification_checker(labels)}')
                loss = data_loss + stratification_epsilon * total_feature_loss / (1 + stratification_checker(labels) * float(1e9)) ** 2
                print(f"iter: {running_iter}, Loss: {loss.item():.4f}, strat: {stratification_checker(labels):.3f}"
                      f"                                    ({(time.time() - start):.3f} s)")

            # Softmax to convert to probabilities
            out = torch.softmax(out, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Name check: Shuffling sanity check
            if i == 0:
                print(f'The test names are: {names[0]}, {names[-2]}')

            # Writing to tensorboard
            # Normalise images
            images = images.cpu().detach().numpy()
            out = out.cpu().detach().numpy()
            images = normalise_image(images)
            out = normalise_image(out)
            labels = labels.cpu().detach().numpy()
            if running_iter % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), running_iter)
                img2tensorboard.add_animated_gif(writer=writer, image_tensor=images[0, ...],
                                                 tag=f'Visuals/Images_Fold_{fold}', max_out=patch_size//2,
                                                 scale_factor=255, global_step=running_iter)
                img2tensorboard.add_animated_gif(writer=writer, image_tensor=labels[0, 0, ...][None, ...],
                                                 tag=f'Visuals/Labels_Fold_{fold}', max_out=patch_size//2,
                                                 scale_factor=255, global_step=running_iter)
                img2tensorboard.add_animated_gif(writer=writer, image_tensor=out[0, 0, ...][None, ...],
                                                 tag=f'Visuals/Output_Fold_{fold}', max_out=patch_size//2,
                                                 scale_factor=255, global_step=running_iter)

            running_iter += 1

        print("Epoch: {}, Loss: {},\n Train Dice: Not implemented".format(epoch, running_loss))

        print('Validation step')
        model.eval()
        val_metric = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=False, softmax=True)
        running_loss = 0
        # correct = 0
        val_counter = 0
        names_collector = []
        metric_collector = []
        gm_volumes_collector = []

        with torch.no_grad():
            for val_sample in val_loader:
                val_images = val_sample['mri']['data'].squeeze().cuda()
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
                val_names = val_sample['mri']['path']
                val_names = [os.path.basename(val_name) for val_name in val_names]

                # Pass images to the model
                if physics_flag:
                    # Calculate physics extensions
                    val_processed_physics = physics_preprocessing(val_physics, physics_experiment_type)
                    out, features_out = model(val_images, val_processed_physics)
                # print(f'Images shape is {images.shape}')
                else:
                    out, features_out = model(val_images)

                val_data_loss = F.binary_cross_entropy_with_logits(out, val_labels, reduction="mean")

                # Loss depends on training mode
                if training_mode == 'standard':
                    val_loss = val_data_loss
                elif training_mode == 'stratification':
                    val_total_feature_loss = 0.1 * calc_feature_loss(
                        features_out)  # NOTE: This needs to be the feature tensor!
                    regulatory_ratio = val_data_loss / val_total_feature_loss
                    val_loss = data_loss + stratification_epsilon * total_feature_loss / (
                                1 + stratification_checker(val_labels) * float(1e9)) ** 2

                # print(f"out val shape is {out.shape}")  # Checking for batch dimension inclusion or not
                out = torch.softmax(out, dim=1)
                gm_out = out[:, 0, ...]

                running_loss += val_loss.item()

                # Metric calculation
                dice_performance = val_metric.forward(out, val_labels)
                gm_volume = gm_out.view(4, -1).sum(1)
                metric_collector += [dice_performance.tolist()]
                names_collector += val_names
                gm_volumes_collector += gm_volume

                # Convert to numpy arrays
                val_images = val_images.cpu().detach().numpy()
                val_labels = val_labels.cpu().detach().numpy()
                val_images = normalise_image(val_images)
                out = out.cpu().detach().numpy()
                out = normalise_image(out)

                val_counter += val_batch_size

        # Write to tensorboard
        writer.add_scalar('Loss/val', running_loss / val_counter, running_iter)
        writer.add_scalar('Loss/dice_val', np.mean(metric_collector) / val_counter, running_iter)
        img2tensorboard.add_animated_gif(writer=writer, image_tensor=val_images[0, ...],
                                         tag=f'Validation/Images_Fold_{fold}', max_out=patch_size // 4,
                                         scale_factor=255, global_step=running_iter)
        img2tensorboard.add_animated_gif(writer=writer, image_tensor=val_labels[0, 0, ...][None, ...],
                                         tag=f'Validation/Labels_Fold_{fold}', max_out=patch_size // 4,
                                         scale_factor=255, global_step=running_iter)
        img2tensorboard.add_animated_gif(writer=writer, image_tensor=out[0, 0, ...][None, ...],
                                         tag=f'Validation/Output_Fold_{fold}', max_out=patch_size // 4,
                                         scale_factor=255, global_step=running_iter)

        # Check if current val dice is better than previous best
        true_dice = np.mean(metric_collector)
        if true_dice > best_val_dice:
            best_val_dice = true_dice
            append_string = 'best'
            best_counter = 0
        else:
            append_string = 'nb'
            best_counter += 1

        # Aggregation
        running_val_metric.append(true_dice)
        running_val_names.append(names_collector)
        running_gm_volumes.append(gm_volumes_collector)

        # Save model
        if SAVE and append_string == 'best':
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
                        'overall_gm_volumes': overall_gm_volumes,
                        'overall_val_names': overall_val_names,
                        'overall_val_metric': overall_val_metric}, MODEL_PATH)

        if best_counter >= 5:
            # Set overalls to best epoch
            best_epoch = int(np.argmin(running_val_metric))
            print(f'The best epoch is Epoch {best_epoch}')
            overall_val_metric.append(running_val_metric[best_epoch])
            overall_val_names.extend(running_val_names[best_epoch])
            overall_gm_volumes.extend(running_gm_volumes[best_epoch])
            break

    # Now that this fold's training has ended, want starting points of next fold to reset
    latest_epoch = -1
    latest_fold = 0
    running_iter = 0

## Totals: What to collect after training has finished
# Dice for all validation? Volumes and COVs?

overall_val_metric = np.array(overall_val_metric)
overall_gm_volumes = np.array(overall_gm_volumes)
overall_subject_ids = [int(vn[0].split('_')[2]) for vn in overall_val_names]

# Folds analysis
print('Names', len(overall_val_names), 'Dice', len(overall_val_metric), 'GM volumes', len(overall_gm_volumes))

# Folds Dice
print('Overall Dice:', np.mean(overall_val_metric), 'std:', np.std(overall_val_metric))

sub = pd.DataFrame({"Filename": overall_val_names,
                    "subject_id": overall_subject_ids,
                    "Dice": overall_val_metric.tolist(),
                    "GM_volumes": overall_gm_volumes})

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
print('Finished!')
