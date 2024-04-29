import pathlib
import sys
import typing


def get_files(
    base_folder: pathlib.Path,
) -> typing.List[typing.Tuple[pathlib.Path, pathlib.Path]]:
    files = []
    images_folder = base_folder.joinpath("images/nii")
    masks_folder = base_folder.joinpath("masks/nii")
    for nii_folder in masks_folder.iterdir():
        mask_filename = nii_folder.joinpath("mask.nii.gz")
        image_filename = images_folder.joinpath(f"{nii_folder.name}/image.nii.gz")
        print(mask_filename, image_filename)
        if image_filename.exists() and mask_filename.exists():
            files.append((image_filename, mask_filename))
    return files


def main():
    base_folder = pathlib.Path(sys.argv[1]).resolve()
    trachea_files = get_trachea_files(base_folder)
    print(len(trachea_files))


if __name__ == "__main__":
    main()
