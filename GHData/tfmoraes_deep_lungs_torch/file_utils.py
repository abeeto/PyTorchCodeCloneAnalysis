import pathlib
import sys
import typing


def get_files(
    base_folder: pathlib.Path,
) -> typing.List[typing.Tuple[pathlib.Path, pathlib.Path]]:
    files = []
    images_folder = base_folder.joinpath("images/")
    masks_folder = base_folder.joinpath("masks/")
    for mask_filename in masks_folder.iterdir():
        image_filename = images_folder.joinpath(f"{mask_filename.name}")
        if image_filename.exists() and mask_filename.exists():
            files.append((image_filename, mask_filename))
    return files


def main():
    base_folder = pathlib.Path(sys.argv[1]).resolve()
    trachea_files = get_files(base_folder)
    print((trachea_files))


if __name__ == "__main__":
    main()
