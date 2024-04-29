import pandas as pd
import os


def sort_dir(dir_name, sub_split=2, param_split=4):
    """
    :param dir_name: Full directory name to be sorted
    :param sub_split: Location of subID to split in sorting
    :param param_split: Location of parameter to split in sorting
    :return: Sorted directory according to first (subID) and third value (TI) following underscores
    """
    # print(dir_name)
    # float(xx.split('_')[param_split].lstrip('0'))])
    sorted_list = sorted(os.listdir(dir_name), key=lambda xx: int(xx.rsplit('.nii.gz')[0].split('_')[sub_split]))
    # sorted_list = sorted(os.listdir(dir_name), key=lambda xx: [int(xx.rsplit('.nii.gz')[0].split('_')[sub_split]),
    #                                                            float(xx.rsplit('.nii.gz')[0].split('_')[param_split])])
    subject_ids = [int(xx.rsplit('.nii.gz')[0].split('_')[sub_split]) for xx in sorted_list]
    subject_physics = [float(xx.rsplit('.nii.gz')[0].split('_')[param_split]) for xx in sorted_list]
    return sorted_list, subject_ids, subject_physics


# Given directory of volumes want to create a csv with 5 folds (training/ valdiation/ inference)
OOD = True
if OOD:
    image_directory = '/data/Resampled_Data/Images/OOD_limited'
    images, subject_ids, subject_physics = sort_dir(image_directory, sub_split=3, param_split=5)
else:
    image_directory = '/data/Resampled_Data/Images/SS_GM_Images'
    images, subject_ids, subject_physics = sort_dir(image_directory, sub_split=2, param_split=4)

shuffle_test = False

basic_csv = pd.DataFrame({
    'subject_id': subject_ids,
    'Filename': images,
    'subject_physics': subject_physics
})


# Fold creation:
def create_csv(dataframe, name='local_physics_csv.csv', folds=True):
    if folds:
        dataframe.fold = 0
        num_subjects = dataframe.subject_id.nunique()
        num_folds = 6
        splits = [5, 10, 15, 19, 23, 27][::-1]
        for fold_num, split in enumerate(splits):
            dataframe.loc[dataframe.subject_id < split, 'fold'] = fold_num
        dataframe = dataframe.astype({'fold': 'int32'})
    
    # Saving "base" csv
    dataframe.to_csv(os.path.join('/home/pedro/PhysicsPyTorch/', name), index=False)


# Default: create_csv(basic_csv)
# OOD
if OOD:
    create_csv(basic_csv, name='OOD_physics_csv_folds_limited.csv', folds=True)
else:
    create_csv(basic_csv, folds=True)

# from sklearn.model_selection import KFold
# kf = KFold(n_splits=num_folds)
# kf.get_n_splits(basic_csv.Filename)
# print(kf)
# for train_index, test_index in kf.split(basic_csv.Filename):
#     # print(train_index[:10])
#     print('This is training index number whatever}')
#     print(basic_csv.Filename[test_index])
#     # print(print(len(train_index), len(test_index)))

# Shuffle batches without shuffling contents of batches: Code taken from own stratified_shuffler.py
import numpy as np
import random
bs = 4


# Not enough to shuffle batches, shuffle WITHIN batches!
# Take original csv, shuffle between subjects!
def reshuffle_csv(og_csv, batch_size):
    # Calculate some necessary variables
    batch_reshuffle_csv = pd.DataFrame({})
    num_images = len(og_csv)
    batch_numbers = list(np.array(range(num_images // batch_size)) * batch_size)
    num_unique_subjects = og_csv.subject_id.nunique()

    # First, re-order within subjects so batches don't always contain same combination of physics parameters
    for sub_ID in range(num_unique_subjects):
        batch_reshuffle_csv = batch_reshuffle_csv.append(og_csv[og_csv.subject_id == sub_ID].sample(frac=1).
                                                         reset_index(drop=True), ignore_index=True)

    # Set up empty lists for appending re-ordered entries
    new_subject_ids = []
    new_filenames = []
    new_physics = []
    new_folds = []
    for batch in range(num_images // batch_size):
        # Randomly sample a batch ID
        batch_id = random.sample(batch_numbers, 1)[0]
        # Find those images/ labels/ params stipulated by the batch ID
        transferred_subject_ids = batch_reshuffle_csv.subject_id[batch_id:batch_id+batch_size]
        transferred_filenames = batch_reshuffle_csv.Filename[batch_id:batch_id+batch_size]
        transferred_physics = batch_reshuffle_csv.subject_physics[batch_id:batch_id+batch_size]
        transferred_folds = batch_reshuffle_csv.fold[batch_id:batch_id+batch_size]
        # Append these to respective lists
        new_subject_ids.extend(transferred_subject_ids)
        new_filenames.extend(transferred_filenames)
        new_physics.extend(transferred_physics)
        new_folds.extend(transferred_folds)
        # Remove batch number used to reshuffle certain batches
        batch_numbers.remove(batch_id)
    
    altered_basic_csv = pd.DataFrame({
        'subject_id': new_subject_ids,
        'Filename': new_filenames,
        'subject_physics': new_physics,
        'fold': new_folds
    })
    return batch_reshuffle_csv, altered_basic_csv


if shuffle_test:
    # Testing the shuffle function
    tester, tester2 = reshuffle_csv(basic_csv, bs)
