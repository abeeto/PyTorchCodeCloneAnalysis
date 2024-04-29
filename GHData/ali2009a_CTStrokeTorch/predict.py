import os

from train import config
from prediction import run_validation_cases


def main():

    prediction_dir = os.path.abspath("data/prediction")
    run_validation_cases(validation_keys_file=None,
                         model_file="/home/aliarab/scratch/outputs/org_wo_ds_128/model_file.h5",
                         labels=config["labels"],
                         hdf5_file= "/home/aliarab/scratch/data/slice_based/ct_test_org_128_64_norm_30_130.h5",
                         output_label_map=True,
                         output_dir=prediction_dir)

if __name__ == "__main__":
    main()

