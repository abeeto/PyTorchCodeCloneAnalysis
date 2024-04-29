import nibabel as nib
import numpy as np
import pathlib
import file_utils

def main():
    dataset_folder = pathlib.Path("datasets/").absolute()
    files = file_utils.get_trachea_files(dataset_folder)
    for n, (image_filename, mask_filename) in enumerate(files):
        mask = nib.load(str(mask_filename)).get_fdata()
        mask = np.array(mask[::-1, :, :])
        nii = nib.Nifti1Image(mask, np.eye(4))
        nib.save(nii, mask_filename)
        print(f"{n}/{len(files)} - {mask_filename.parent} - {mask.shape}")

if __name__ == "__main__":
    main()
