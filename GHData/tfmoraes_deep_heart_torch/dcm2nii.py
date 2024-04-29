import argparse
import pathlib
import sys

import gdcm
import imageio
import nibabel as nib
import numpy as np
import pydicom

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i",
    "--input",
    type=pathlib.Path,
    metavar="folder",
    help="Dicom input folder",
    dest="input_folder",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    default="nii",
    type=pathlib.Path,
    metavar="folder",
    help="output folder",
    dest="output_folder",
    required=True,
)

ignore = ["ID00052637202186188008618"]


def get_gdcm_to_numpy_typemap():
    """Returns the GDCM Pixel Format to numpy array type mapping."""
    _gdcm_np = {
        gdcm.PixelFormat.UINT8: np.uint8,
        gdcm.PixelFormat.INT8: np.int8,
        # gdcm.PixelFormat.UINT12 :np.uint12,
        # gdcm.PixelFormat.INT12  :np.int12,
        gdcm.PixelFormat.UINT16: np.uint16,
        gdcm.PixelFormat.INT16: np.int16,
        gdcm.PixelFormat.UINT32: np.uint32,
        gdcm.PixelFormat.INT32: np.int32,
        # gdcm.PixelFormat.FLOAT16:np.float16,
        gdcm.PixelFormat.FLOAT32: np.float32,
        gdcm.PixelFormat.FLOAT64: np.float64,
    }
    return _gdcm_np


def get_numpy_array_type(gdcm_pixel_format):
    """Returns a numpy array typecode given a GDCM Pixel Format."""
    return get_gdcm_to_numpy_typemap()[gdcm_pixel_format]


# Based on http://gdcm.sourceforge.net/html/ConvertNumpy_8py-example.html
def gdcm_to_numpy(filename, apply_intercep_scale=False):
    reader = gdcm.ImageReader()
    reader.SetFileName(filename)
    if not reader.Read():
        raise Exception(f"It was not possible to read {filename}")
    image = reader.GetImage()
    pf = image.GetPixelFormat()
    if image.GetNumberOfDimensions() == 3:
        shape = (
            image.GetDimension(2),
            image.GetDimension(1),
            image.GetDimension(0),
            pf.GetSamplesPerPixel(),
        )
    else:
        shape = image.GetDimension(1), image.GetDimension(0), pf.GetSamplesPerPixel()
    dtype = get_numpy_array_type(pf.GetScalarType())
    gdcm_array = image.GetBuffer()
    np_array = np.frombuffer(
        gdcm_array.encode("utf-8", errors="surrogateescape"), dtype=dtype
    )
    np_array.shape = shape
    np_array = np_array.squeeze()

    if apply_intercep_scale:
        shift = image.GetIntercept()
        scale = image.GetSlope()
        output = np.empty_like(np_array, np.int16)
        output[:] = scale * np_array + shift
        return output
    else:
        return np_array


def read_dicom_to_ndarray(folder: pathlib.Path) -> np.ndarray:
    print(folder)
    dicom_arrs = []
    dicom_files = list([str(i) for i in folder.iterdir()])
    sorter = gdcm.IPPSorter()
    sorter.Sort(dicom_files)
    sorted_dicom_files = sorter.GetFilenames()
    if len(sorted_dicom_files) != len(dicom_files):
        sorted_dicom_files = dicom_files
    for dcm_file in sorted_dicom_files:
        dicom_arrs.append(gdcm_to_numpy(dcm_file))
    return np.array(dicom_arrs)


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

    for dicom_folder in input_folder.iterdir():
        if dicom_folder.name not in ignore:
            dcm_array = read_dicom_to_ndarray(dicom_folder)
            nii_filename = output_folder.joinpath(dicom_folder.name, "image.nii.gz")
            nii_filename.parent.mkdir(parents=True, exist_ok=True)
            save_to_nii(dcm_array, str(nii_filename))
            # create_screenshot(dcm_array, str(nii_filename).replace("nii.gz", "png"))


if __name__ == "__main__":
    main()
