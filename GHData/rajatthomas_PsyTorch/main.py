from utils import create_patch_files

if __name__ == "__main__":

    # Create MRI patches from original data
    root_dir = '/data_local/deeplearning/data'
    tot_patches = create_patch_files(root_dir=root_dir)

    print(tot_patches)