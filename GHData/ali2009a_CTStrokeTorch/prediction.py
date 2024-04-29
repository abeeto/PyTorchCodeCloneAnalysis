import os

import nibabel as nib
import numpy as np
import tables
import time
from training import load_old_model
from data  import pickle_load
#from .utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices
from monai.transforms import Resize

def reconstruct_from_slices(output_shape, slices):
    data = np.ones(output_shape)
    for index,s in enumerate(slices):
        data[:,:,index] = s[0,...]
    return data

def get_slice_from_3d_data(data, slice_index):
    return data[...,slice_index]


def slice_based_prediction(model, data, batch_size=1):
    #data shape is (1, 1, 256, 256, 64)
    images_depth = data.shape[-1]
    predictions = list()
    indices = list(range(images_depth))
    batch = list()
    i = 0
    while i < len(indices):
        while len(batch) < batch_size:
            slice = get_slice_from_3d_data(data[0], slice_index=indices[i])
            batch.append(slice)
            i += 1
        prediction = predict(model, np.asarray(batch))
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape =  list(data.shape[-3:])
    return reconstruct_from_slices(output_shape, predictions)


def get_test_indices(testing_file):
    return pickle_load(testing_file)


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


def prediction_to_image(prediction, affine, label_map=True, threshold=0.5, labels=None):
    data = prediction[0, 0]
    if label_map:
        label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
        if labels:
            label = labels[0]
        else:
            label = 1
        label_map_data[data > threshold] = label
        data = label_map_data
    return nib.Nifti1Image(data, affine)

def resize_to_original_size(image, original_size):
    original_size = [int(dim) for dim in original_size]
    resize = Resize(original_size, mode="nearest")
    resized_data = resize(image.get_fdata()[np.newaxis])
    resized_img = nib.Nifti1Image(resized_data[0], image.affine)
    return resized_img

def run_validation_case(data_index, output_dir, model, data_file,
                        output_label_map=False, threshold=0.5, labels=None, batch_size=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    original_size=  data_file.root.size[data_index]
    print (original_size)
    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])
    
    image = nib.Nifti1Image(test_data[0, 0], affine)
    image = resize_to_original_size(image, original_size)
    image.to_filename(os.path.join(output_dir, "data_ct.nii.gz"))

    test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
    test_truth = resize_to_original_size(test_truth, original_size)
    test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))

    
    input_shape  = tuple([int(dim) for dim in model.input.shape[1:]]) # (1, 256, 256) for slices based
    #print ("before precition\n")
    print("test date shape:")
    print(test_data.shape)
    print ("input shape:")
    print(input_shape)
    if len(input_shape) == len(test_data.shape)-1:
        prediction = predict(model, test_data) #used np to add a new axis, it might be wrong
    else:
        #add new axis for batch dimension
        prediction = slice_based_prediction(model=model, data=test_data, batch_size=batch_size)[np.newaxis][np.newaxis]
    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                           labels=labels)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        prediction_image = resize_to_original_size(prediction_image, original_size)
        prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))


def run_validation_cases(validation_keys_file, model_file, labels, hdf5_file,
                         output_label_map=False, output_dir=".", threshold=0.5, batch_size=1):
    data_file = tables.open_file(hdf5_file, "r")
    if validation_keys_file:
        validation_indices = pickle_load(validation_keys_file)
        validation_indices=validation_indices[:4]
    else:
        sample_num= len(data_file.root.data)
        validation_indices = list(range(sample_num))
    model = load_old_model(model_file)
    for index in validation_indices:
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        start = time.time()
        print("predicting for case: {}".format(index))
        run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            output_label_map=output_label_map, labels=labels,
                            threshold=threshold, batch_size=batch_size)
        end = time.time()
        print ("time:",end - start)
        
    data_file.close()



def predict(model, data):
    print("shape:")
    print(data.shape)
    print (model.input.shape)
    return model.predict(data)



