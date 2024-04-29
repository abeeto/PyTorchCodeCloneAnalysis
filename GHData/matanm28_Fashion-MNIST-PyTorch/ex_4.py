import concurrent
import sys
import matplotlib.pyplot as plt
from concurrent.futures.thread import ThreadPoolExecutor

import torch
from numpy import ndarray
from torchvision import transforms, datasets
import numpy as np
from typing import Dict

from fashion_dataset import StdNormalizer, FashionDataset
from model_a import ModelA
from model_b import ModelB
from model_c import ModelC
from model_d import ModelD
from model_e import ModelE
from model_f import ModelF
from model_wrapper import ModelWrapper

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:6]

MODELS = {
    'A': ModelA,
    'B': ModelB,
    'C': ModelC,
    'D': ModelD,
    'E': ModelE,
    'F': ModelF,
}

HYPER_PARAMS = {
    'A': {'lr': 0.01, 'batch_size': 64, 'epochs': 12},
    'B': {'lr': 0.01, 'batch_size': 80, 'epochs': 12},
    'C': {'lr': 0.001, 'batch_size': 64, 'epochs': 12},
    'D': {'lr': 0.01, 'batch_size': 64, 'epochs': 12},
    'E': {'lr': 0.01, 'batch_size': 64, 'epochs': 12},
    'F': {'lr': 0.01, 'batch_size': 250, 'epochs': 12},
}

MODELS_ADDITIONAL_PARAMS = {
    'C': {'dropout': [0.5, 0.25]}
}

FILE_NAME = 'test_y'


def normalize_data_std(data_set: ndarray):
    mean = data_set.mean(axis=0)
    std_dev = data_set.std(axis=0)
    return np.nan_to_num((data_set - mean) / std_dev)


def normalize_data_min_max(data_set: ndarray, new_min: int = 0, new_max: int = 1):
    data_min = data_set.min(axis=0)
    data_max = data_set.max(axis=0)
    return ((data_set - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min


def run_model(training_set: FashionDataset, validation_set: FashionDataset, test_set: FashionDataset,
              model_name: str, epochs: int, lr: float, batch_size: int):
    print(f'Running with model {model_name}')
    if model_name in MODELS_ADDITIONAL_PARAMS.keys():
        model = MODELS[model_name](training_set.data.shape[1], lr, MODELS_ADDITIONAL_PARAMS[model_name])
    else:
        model = MODELS[model_name](training_set.data.shape[1], lr)
    model_wrapper = ModelWrapper(model, epochs, batch_size, model_name)

    training_loss_arr, training_accuracy_arr = model_wrapper.train(training_set)
    validation_loss, validation_accuracy = model_wrapper.test(validation_set)
    test_loss, test_accuracy = model_wrapper.test(test_set)
    return model_wrapper, (training_loss_arr, training_accuracy_arr), (validation_loss, validation_accuracy), (
        test_loss, test_accuracy)


def load_test_set(test_x_path):
    test_set, _ = FashionDataset.from_files(test_x_path)
    test_transform_func = transforms.Compose([StdNormalizer(test_set.data)])
    test_set.set_transforms(test_transform_func)
    return test_set


def write_predictions_to_file(file_name: str, predictions: ndarray):
    with open(file_name, 'w') as file:
        file.write('\n'.join([str(i) for i in predictions.tolist()]))


def get_best_model_with_threads(train_x_path: str, train_y_path: str, num_of_rows: int = None):
    models = {}
    training_set, validation_set, online_validation_set = load_fashion_datasets(num_of_rows, train_x_path, train_y_path)
    futures = {}
    with ThreadPoolExecutor(6) as executor:
        for name in MODELS:
            params = HYPER_PARAMS[name]
            futures[name] = executor.submit(run_model, training_set, validation_set, online_validation_set, name,
                                            params['epochs'], params['lr'], params['batch_size'])

        for future in concurrent.futures.as_completed(futures.values()):
            model_wrapper, training, validation, test = future.result()
            name = model_wrapper.model_name
            torch.save(model_wrapper.model.state_dict(), f'saved_models/{name}.pt')
            models[name] = {'model': model_wrapper, 'training': training, 'validation': validation, 'test': test}
            print(f'Model {name}:')
            print(f'loss on validation: {validation[0]}, accuracy on validation: {validation[1]}%')
            print(f'loss on test: {test[0]}, accuracy on test: {test[1]}%')
    best_model_name = 'A'
    best_avg = (models[best_model_name]['validation'][1] + models[best_model_name]['test'][1]) / 2
    for name in models:
        if name == best_model_name:
            continue
        curr_avg = (models[name]['validation'][1] + models[name]['test'][1]) / 2
        if curr_avg > best_avg:
            best_model_name = name
            best_avg = curr_avg
    return models, best_model_name


def load_fashion_datasets(num_of_rows, train_x_path, train_y_path):
    training_set, validation_set = FashionDataset.from_files(train_x_path, train_y_path, num_of_rows, 0.2)
    train_transform_func = transforms.Compose([StdNormalizer(training_set.data)])
    training_set.set_transforms(train_transform_func)
    validation_set_transforms = transforms.Compose([StdNormalizer(validation_set.data)])
    validation_set.set_transforms(validation_set_transforms)
    test_dataset = datasets.FashionMNIST('.\data', train=False, download=True)
    flat_test_tensor = test_dataset.data.reshape((test_dataset.data.shape[0], 784)).type(torch.float32)
    test_set_transforms = transforms.Compose([StdNormalizer(flat_test_tensor)])
    test_set = FashionDataset(flat_test_tensor.numpy(), test_dataset.targets.numpy())
    test_set.set_transforms(test_set_transforms)
    return training_set, validation_set, test_set


def plot_data(models_data: Dict):
    model_list = list(MODELS.keys())
    loss_training = [models_data[name]['training'][0] for name in model_list]
    accuracy_training = [models_data[name]['training'][1] for name in model_list]
    loss_validation = [models_data[name]['validation'][0] for name in model_list]
    accuracy_validation = [models_data[name]['validation'][1] for name in model_list]
    mode = ['loss', 'accuracy']
    train = [loss_training, accuracy_training]
    vali = [loss_validation, accuracy_validation]
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), dpi=500)
    axs = axs.flatten()
    fig.suptitle('Results', fontsize=20)
    # validation graphs
    for i in range(2):
        axs[i].set_title(f'$Training$ ${mode[i]}$')
        axs[i].set_xlabel('$Iterations$')
        axs[i].set_ylabel(f'$Average$ ${mode[i]}$')
        for model_index, color in zip(range(len(model_list)), colors):
            axs[i].plot(train[i][model_index], c=color, lw=2, label=model_list[model_index])
        axs[i].legend(ncol=3, fontsize='large', loc='upper right', fancybox=True, framealpha=0.5)
    for i in range(2, 4):
        axs[i].set_title(f'$Validation$ ${mode[i % 2]}$')
        axs[i].bar(model_list, vali[i % 2])
    plt.savefig('plot.jpeg')


def generate_report(train_x_path: str, train_y_path: str, test_x_path: str, num_of_rows: int = None):
    models_data, best_model_name = get_best_model_with_threads(train_x_path, train_y_path, num_of_rows)
    print(f'Best model is {best_model_name}')
    plot_data(models_data)
    test_data = load_test_set(test_x_path)
    predictions = models_data[best_model_name]['model'].predict(test_data)
    write_predictions_to_file(FILE_NAME, predictions.numpy())


def predict_with_best_model(train_x_path: str, train_y_path: str, test_x_path: str, best_model: str,
                            num_of_rows: int = None):
    training_set, validation_set, test_set = load_fashion_datasets(num_of_rows, train_x_path, train_y_path)
    hyper_params = HYPER_PARAMS[best_model]
    model_wrapper, _, validation, test = run_model(training_set, validation_set, test_set, best_model,
                                                   hyper_params['epochs'], hyper_params['lr'],
                                                   hyper_params['batch_size'])
    print(f'Model {best_model}:')
    print(f'loss on validation: {validation[0]}, accuracy on validation: {validation[1]}%')
    print(f'loss on test: {test[0]}, accuracy on test: {test[1]}%')
    test_data = load_test_set(test_x_path)
    predictions = model_wrapper.predict(test_data)
    write_predictions_to_file(FILE_NAME, predictions.numpy())


if __name__ == '__main__':
    LABELS = {
        0: 'T-shirt/top',
        1: 'Trousers',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    NUM_OF_ROWS = None
    BEST_MODEL = 'D'
    if len(sys.argv) < 4:
        predict_with_best_model('train_x.txt', 'train_y.txt', 'test_x.txt', BEST_MODEL)
    elif len(sys.argv) == 4:
        predict_with_best_model(sys.argv[1], sys.argv[2], sys.argv[3], BEST_MODEL)
    elif len(sys.argv) == 5:
        generate_report(sys.argv[1], sys.argv[2], sys.argv[3])
