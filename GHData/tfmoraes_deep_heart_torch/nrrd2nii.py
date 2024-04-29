import argparse
import pathlib
import sys

import imageio
import nibabel as nib
import nrrd
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i",
    "--input",
    type=pathlib.Path,
    metavar="folder",
    help="NRRD input folder",
    dest="input_folder",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    default="output.vti",
    type=pathlib.Path,
    metavar="folder",
    help="output folder",
    dest="output_folder",
    required=True,
)

ignore = ["ID00052637202186188008618"]


def read_nrrd_to_ndarray(filename: pathlib.Path) -> np.ndarray:
    _nrrd = nrrd.read(filename)
    arr = np.array(np.swapaxes(_nrrd[0], 0, 2))
    return arr


def save_to_nii(data: np.ndarray, filename: str):
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filename)


def create_screenshot(data: np.ndarray, filename: str):
    mip_image = data.max(0)
    imageio.imwrite(filename, mip_image)


def main():
    args, _ = parser.parse_known_args()
    input_folder = args.input_folder.absolute()
    output_folder = args.output_folder.absolute()
    for nrrd_filename in input_folder.iterdir():
        data = read_nrrd_to_ndarray(nrrd_filename)
        nii_filename = output_folder.joinpath(nrrd_filename.name.split('_')[0], "mask.nii.gz")
        nii_filename.parent.mkdir(parents=True, exist_ok=True)
        save_to_nii(data, str(nii_filename))
        # create_screenshot(dcm_array, str(nii_filename).replace("nii.gz", "png"))


if __name__ == "__main__":
    main()
