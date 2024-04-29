import argparse
import pathlib
import sys

import imageio
import nibabel as nib
import numpy as np
import pydicom

ignore = ["ID00132637202222178761324"]

def read_dicom_to_ndarray(folder: pathlib.Path) -> np.ndarray:
    dicom_arrs = []
    for dcm_file in sorted(folder.iterdir(), key=lambda x: int(x.name.split(".")[0])):
        dicom_arrs.append(pydicom.dcmread(dcm_file).pixel_array)
    return np.array(dicom_arrs)


def read_images_to_ndarray(folder: pathlib.Path) -> np.ndarray:
    image_files = list(folder.iterdir())
    image_files.sort(key=lambda f: int(f.name.split('.')[0]))
    images = []
    for image_file in image_files:
        image = imageio.imread(image_file)
        images.append(image)
    return np.array(images)


def save_to_nii(data: np.ndarray, filename: pathlib.Path):
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filename)


def create_screenshot(data: np.ndarray, filename: str):
    mip_image = data.max(0)
    imageio.imwrite(filename, mip_image)



def main():
    masks_folder = pathlib.Path('datasets/mask_clear/mask_clear/')

    images_train_folder = pathlib.Path('datasets/train')
    images_test_folder = pathlib.Path('datasets/test')

    images_output_folder = pathlib.Path('datasets/images')
    masks_output_folder = pathlib.Path('datasets/masks')

    images_output_folder.mkdir(parents=True, exist_ok=True)
    masks_output_folder.mkdir(parents=True, exist_ok=True)

    for mask_folder in masks_folder.iterdir():
        if mask_folder.name not in ignore:
            if images_train_folder.joinpath(mask_folder.name).exists():
                image_folder = images_train_folder.joinpath(mask_folder.name)
            elif images_test_folder.joinpath(mask_folder.name).exists():
                image_folder = images_test_folder.joinpath(mask_folder.name)
            else:
                print('err')
                continue
            print(image_folder)
            dcm_array  = read_dicom_to_ndarray(image_folder)
            images_array = read_images_to_ndarray(mask_folder)
            save_to_nii(dcm_array, images_output_folder.joinpath(mask_folder.name).with_suffix('.nii.gz'))
            save_to_nii(images_array, masks_output_folder.joinpath(mask_folder.name).with_suffix('.nii.gz'))
        # if dicom_folder.name not in ignore:
        #     dcm_array = read_dicom_to_ndarray(dicom_folder)
        #     nii_filename = output_folder.joinpath(dicom_folder.name, "image.nii.gz")
        #     nii_filename.parent.mkdir(parents=True, exist_ok=True)
        #     save_to_nii(dcm_array, str(nii_filename))
        #     #create_screenshot(dcm_array, str(nii_filename).replace("nii.gz", "png"))


if __name__ == "__main__":
    main()
