import os
import shutil


def copyfiles(copy_path, dstpath, file_ext=None):
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    for file_path in os.listdir(copy_path):
        if file_ext is None:
            shutil.copyfile(os.path.join(copy_path, file_path), os.path.join(dstpath, file_path))
        else:
            if file_path.endswith(file_ext):
                shutil.copyfile(os.path.join(copy_path, file_path), os.path.join(dstpath, file_path))


def prepare_data():
    source_folder1 = r"d:\download\bounding_box_train_camstyle_duke\bounding_box_train_camstyle_duke"
    source_folder2 = r"data/bounding_box_train"
    des_folder = r"data\duke_merge"
    # des_folder = r"D:\ANewspace\code\JVTC_ms\data"
    copyfiles(source_folder1, des_folder)
    copyfiles(source_folder2, des_folder)


def unzipping(path_to_zip_file, directory_to_extract_to):
    import zipfile

    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def mvoin_files_rubb():
    src_1 = r"data/duke"
    dest_1 = r"data/duke_merge"
    copyfiles(src_1, dest_1, file_ext=".jpg")


if __name__ == '__main__':
    # prepare_data()
    # unzipping()
    mvoin_files_rubb()
    # unzipping("data/ljkkjkjjjk.zip", "data/duke")
