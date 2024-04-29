import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob


def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``
    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin


def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)
    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)

def reorientation(source_path, save_path):
    source_nii = nib.load(source_path)
    source_data = source_nii.get_data()  # shape = (512, 512, 123)
    source_axcodes = tuple(nib.aff2axcodes(source_nii.affine))  # ('R', 'A', 'S')
    print(source_data.shape)
    target_axcodes = ('I','L','P')  # ('I', 'P', 'L')

    new_img = do_reorientation(source_data, source_axcodes, target_axcodes)  # shape = (523, 523, 100)
    print(new_img.shape)

    new_nii = nib.Nifti1Image(new_img, source_nii.affine, source_nii.header)

    nib.save(new_nii, save_path)

source = os.path.join('ixi_tiny','IXI002-Guys-0828_image.nii.gz')
save_path = 'IXI002-Guys-0828_image.nii.gz'
reorientation(source,save_path)