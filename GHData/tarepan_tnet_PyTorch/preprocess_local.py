from StarGAN_VC.preprocesses.preprocess_run import run_preprocess

from pathlib import Path


refresh = True
mode = None

resampling_rate = 16000
root_dir = Path("./processed_data")
datasets = [
    # "a": {
    #     "zip_name": "BASIC5000",
    #     "zip_path": Path("../datasets/official/JSUT_basic5000")/"BASIC5000.zip",
    #     "file_base": "BASIC5000_",
    #     "train_range": [1, 4997],
    #     "eval_range": [4998, 5000],
    #     "file_digit": 4
    # },
    {
        "zip_name": "tsuchiya_normal",
        "zip_path": Path("../datasets/official/tsuchiya_5")/"tsuchiya_normal.zip",
        "file_base": "tsuchiya_normal_",
        "train_range": [1, 3], # simply [start, end], not [start, end)
        "eval_range": [4, 5],
        "file_digit": 3
    },{
        "zip_name": "hiroshiba_normal",
        "zip_path": Path("../datasets/official/nico_hiho5")/"hiroshiba_normal.zip",
        "file_base": "hiroshiba_normal_",
        "train_range": [4, 5], # simply [start, end], not [start, end)
        "eval_range": [1, 3],
        "file_digit": 3
    },
    # "c": {
    #     "zip_name": "hiroshiba_normal",
    #     "zip_path": Path("../datasets/official/nico_hiho100")/"hiroshiba_normal.zip",
    #     "file_base": "hiroshiba_normal_",
    #     "train_range": [1, 97], # simply [start, end], not [start, end)
    #     "eval_range": [98, 100],
    #     "file_digit": 3
    # }
]

run_preprocess(root_dir, refresh, datasets, resampling_rate, mode)
