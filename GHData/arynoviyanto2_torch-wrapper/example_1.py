from two_dimension_dataset import TwoDimensionDataset
from helpers import generate_metadata_file, load_model
from cnn_model import CnnModel
from resnet_model import ResnetModel
from ml_model import MlModel
from engine import Engine
from performance_evaluation import Metric

# Construct dataset
dataset_name = 'hand_signs_dataset'
params = { 'batch_size': 10 }

# Training set
train_dir = 'train_signs'
generate_metadata_file(f'{dataset_name}/{train_dir}')
train_dataset = TwoDimensionDataset(f'{dataset_name}/{train_dir}')
train_dataloaders, test_dataloaders = train_dataset.getDataLoaders(params)
#print(train_dataloaders)

_, num_targets = train_dataset.getTargets()

# Validation set
val_dir = 'test_signs'
generate_metadata_file(f'{dataset_name}/{val_dir}')
val_dataset = TwoDimensionDataset(f'{dataset_name}/{val_dir}', nFold=1)
val_dataloader = val_dataset.getDataLoader(params)

model_name = 'resnet'

def getModel(model_name, num_targets, dataset_name, fold=None):
    name = f'{dataset_name}_{fold}' if fold is not None else f'{dataset_name}'

    model = CnnModel(num_targets, name) # by default it is CNN
    if model_name == 'resnet':
        model = ResnetModel(num_targets, name)

    return model

for fold in range(1):
    print('Fold {}'.format(fold))

    # ML model
    ml_model = MlModel(getModel(model_name, num_targets, dataset_name, fold))

    # Engine
    params = {
        'epochs': 3,
        'storage_dir': 'models'
        }
    engine = Engine(ml_model=ml_model, params=params)

    engine.run_training(train_dataloaders[fold], test_dataloaders[fold])

    # Validation
    best_model = getModel(model_name, num_targets, dataset_name)
    model_name = ml_model.manifest().get_name()
    storage_dir = params['storage_dir']
    load_model(model_name, storage_dir, best_model)

    pe = engine.evaluate(val_dataloader, best_model)
    metrics = pe.get_performance_metrics()
    accuracy = metrics[Metric.ACCURACY]
    sens = metrics[Metric.SENSITIVITY]
    spec = metrics[Metric.SPECITIFITY]
    bal_acc = metrics[Metric.BALANCED_ACCURACY]
    cm = metrics[Metric.CM]

    print(f'===> Fold: {fold}, Accuracy: {accuracy:.2f}')
    print(f'===> Fold: {fold}, sens: {sens:.2f}')
    print(f'===> Fold: {fold}, spec: {spec:.2f}')
    print(f'===> Fold: {fold}, bal_acc: {bal_acc:.2f}')
    print(cm)

