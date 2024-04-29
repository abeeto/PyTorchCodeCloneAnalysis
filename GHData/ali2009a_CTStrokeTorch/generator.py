from random import shuffle
from data import pickle_dump, pickle_load
import data
import itertools
import numpy as np
import os
from tqdm import tqdm

def get_training_and_validation_generators(data_file, batch_size=4, data_split=0.8, validation_keys_file="val_keys.pkl", training_keys_file="train_keys.pkl", slice_based=True, validation_batch_size=4, skip_blank=False):
    print("generating the training/validation split")
    training_list, validation_list = get_validation_split(data_file, 
                                                          training_keys_file=training_keys_file,  
                                                          validation_keys_file=validation_keys_file,
                                                          data_split = data_split)

    print("creating the generators...")
    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size,
                                        slice_based = slice_based,
                                        skip_blank=skip_blank)
    validation_generator = data_generator(data_file, validation_list,
                                        batch_size=batch_size,
                                        slice_based=slice_based,
                                        skip_blank=skip_blank)
    print ("computing the #training/validation steps...")
    num_training_steps = get_number_of_steps(get_number_of_instances(data_file, training_list.copy(), slice_based, skip_blank), batch_size)
    num_validation_steps = get_number_of_steps(get_number_of_instances(data_file, validation_list.copy(), slice_based, skip_blank), batch_size)
    return [training_generator, validation_generator, num_training_steps, num_validation_steps]                                        

def data_generator(data_file, index_list, batch_size, slice_based=True, skip_blank=True):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        if slice_based:
            z = data_file.root.data.shape[-1]
            index_list = create_slice_index_list(orig_index_list, z)
        else:
            index_list = copy.copy(orig_index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, slice_based, skip_blank)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield np.asarray(x_list), np.asarray(y_list)
                x_list = list()
                y_list = list()

def get_number_of_instances(data_file, index_list, slice_based=True, skip_blank=True):
    if slice_based:
        z = data_file.root.data.shape[-1]
        index_list = create_slice_index_list(index_list,z)
        count = 0
        for index in tqdm(index_list):
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, slice_based, skip_blank)
            if len(x_list) > 0:
                count += 1
        return count
    else:
        return len(index_list)



def create_slice_index_list(index_list, z):
    slice_index = list()
    for index in index_list:
        slice_nums = list(range(z))
        slice_index.extend(itertools.product([index], slice_nums))
    return slice_index


def add_data(x_list, y_list, data_file, index, slice_based=None, skip_blank=False):
    data, truth = get_data_from_file(data_file, index, slice_based=slice_based)
    truth = truth[np.newaxis]
    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)



def get_data_from_file(data_file, index, slice_based=None):
    if slice_based:
        index, slice_index = index
        data, truth = get_data_from_file(data_file, index, slice_based=False)
        x= data[:,:,:,slice_index]
        y= truth[:,:,slice_index]
    else:
        x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y




def get_validation_split(data_file, training_keys_file, validation_keys_file, data_split=0.8):
    if not os.path.exists(training_keys_file):
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_keys_file)
        pickle_dump(validation_list, validation_keys_file)
        return training_list, validation_list
    else:
        return  pickle_load(training_keys_file), pickle_load(validation_keys_file)


def split_list(input_list, split=0.8):
    shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1

    




