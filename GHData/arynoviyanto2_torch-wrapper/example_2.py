from two_dimension_dataset import TwoDimensionDataset
from helpers import generate_metadata_file, load_model
from cnn_model import CnnModel
from ml_model import MlModel
from engine import Engine
from performance_evaluation import Metric

# Construct dataset
dataset_name = 'hand_signs_dataset'
params = { 'batch_size': 10 }


# Sample set
val_dir = 'test_signs'
generate_metadata_file(f'{dataset_name}/{val_dir}')
val_dataset = TwoDimensionDataset(f'{dataset_name}/{val_dir}', nFold=1)
val_dataset.sample()
val_dataset.sample(row_index_list=[1, 2])