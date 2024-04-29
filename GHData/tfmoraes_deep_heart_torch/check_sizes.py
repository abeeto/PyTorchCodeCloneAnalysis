import pathlib
import nibabel as nib
import numpy as np
import shutil

import sys

def read_nii(filename: pathlib.Path) -> np.ndarray:
    nii = nib.load(str(filename))
    return nii.get_fdata()

def main():
    masks_folder = pathlib.Path("datasets/masks/nii/").absolute()
    images_folder = pathlib.Path("datasets/images/nii/").absolute()
    for nii_folder in masks_folder.iterdir():
        mask_filename = nii_folder.joinpath("mask.nii.gz")
        image_filename = images_folder.joinpath(nii_folder.name).joinpath("image.nii.gz")

        if image_filename.exists() and mask_filename.exists():
            image = read_nii(image_filename)
            mask = read_nii(mask_filename)
            print(f"{image.shape=}, {mask.shape=}")
            assert(np.all(image.shape == mask.shape))
            # if image.shape == mask.shape:
            #     shutil.copytree(image_filename.parent, output_folder.joinpath(image_filename.parent.name))



if __name__ == "__main__":
    main()
