# стандартные модули
import os.path
import time
from datetime import timedelta, datetime
from pathlib import Path
from collections import namedtuple

# импорт модулей pytorch
import torch
import torch.nn as nn

import torch.optim as optim
# from torch.autograd import Variable

import graphs_shower

from networks.simple_emnist_cnn import SimpleEmnistConvNet
from networks.simple_mnist_cnn import SimpleMnistConvNet
from networks.simple_emnist_ffn import SimpleEmnistFeedForward
from networks.simple_mnist_ffn import SimpleMnistFeedForward
from networks.emnist_cnn_with_bn import EmnistConvBnDropNet


from network_testing import test_net
from network_training import train_net
# from data_loaders import DataLoader
from datasets.emnist_loader import EmnistLoader
from datasets.mnist_loader import MnistLoader


# Сохраняет статистику в файл
def save_stats(accuracies, losses, common_time, save_file):
    """
    Save statistic to file

    :param accuracies:

    :param losses:

    :param common_time: time in format (train,test)

    :param save_file: filename for save file

    :return:
    """
    with open(save_file, "w+") as file:
        file.write("Train:{}\nTest:{}\n".format(common_time[0], common_time[1]))
        epochs_count = len(accuracies)
        for i in range(0, epochs_count):
            file.write("{}:{}\n".format(accuracies[i], losses[i]))


def execute_with_time_measure(func, *parameters):
    """

    :param func: function to measure

    :type func: function

    :param parameters: function parameters

    """
    start_time = time.time()
    func_result = func(*parameters)
    test_time = time.time() - start_time
    delta = timedelta(seconds=round(test_time))
    time_str = str(delta)
    print(f"{func.__name__}: {delta} secs (Wall clock time)")

    return func_result, time_str


# функции потерь на каждой эпохе
epoch_losses = list()
# список значений точности на каждой эпохе
acc_list = list()

NETWORK_TYPE = "CNN"

# Назначаем устройство на котором будет работать нейросеть, по возможности CUDA
dev = "cuda" if torch.cuda.is_available() else "cpu"
used_device = torch.device(dev)
print(f"Running on Device:{used_device}")

Network_Model = namedtuple("Network_Model", ["name", "init"])
MODELS = {
    "0": Network_Model("simple_ffn_mnist", SimpleMnistFeedForward),
    "1": Network_Model("simple_ffn_emnist", SimpleEmnistFeedForward),
    "2": Network_Model("simple_cnn_mnist", SimpleMnistConvNet),
    "3": Network_Model("simple_cnn_emnist", SimpleEmnistConvNet),
    "4": Network_Model("emnist_cnn_with_bn", EmnistConvBnDropNet),
}

# создаём модель
# передаём вычисления на нужное устройство gpu/cpu

print("Модели")
for key_index, model in MODELS.items():
    print(f"{key_index}. {model.name}")
input_model = input("Выберите модель:")
model_name, key_model = ((MODELS[input_model].name, input_model)
                         if input_model in MODELS
                         else ("simple_ffn_mnist", input_model))
chosen_model = MODELS[key_model].init

print("Загружаем модель .....")
net_model = chosen_model(used_device).to(used_device)
# print(net_model)

# проверяем есть ли сохранённая модель
# model_path = r".\models\CNN_EMNIST_model"
model_path = Path(r".\models\test") / model_name
# is_model_exists = os.path.isfile(model_path)
# print(model_name, model_path)
if model_path.exists():
    net_model.load_state_dict(torch.load(model_path))
    net_model.eval()

# скорость обучения
learning_rate = 0.001
OPTIMIZERS = {
    "SGD": optim.SGD(net_model.parameters(), lr=learning_rate, momentum=0.9),
    "Adagrad": optim.Adagrad(net_model.parameters(), lr=learning_rate),
    "Adam": optim.Adam(net_model.parameters(), lr=learning_rate),
    "AdamBetas": optim.Adam(net_model.parameters(), lr=learning_rate, betas=(0.2, 0.01)),
}

print("Оптимизаторы")
for index, optimizer in enumerate(OPTIMIZERS.keys()):
    print(f"{index}. {optimizer}")
input_optimizer = input("Выберите оптимизатор(по умолчанию SGD):")
optimizer_name = input_optimizer if input_optimizer in OPTIMIZERS else "SGD"
optimizer = OPTIMIZERS[optimizer_name]

Dataset = namedtuple("Dataset", ["name", "loader", "path"],
                     defaults=[None, None, None])
DATASETS = {
    "0": Dataset("MNIST", MnistLoader, r"C:\Users\IVAN\Desktop\dataMNIST"),
    "1": Dataset("EMNIST", EmnistLoader, r"C:\Users\IVAN\Desktop\dataEMNIST")
}

print("Датасеты")
for key, dataset in DATASETS.items():
    print(f"{key}. {dataset.name}")
input_dataset = input("Выберите датасет(по умолчанию MNIST):")

dataset_key = (input_dataset if input_dataset in DATASETS
               else "0")

dataset_loader = DATASETS[dataset_key].loader


# str_path = input("Путь до датасета(если путь некоректный выбирается по умолчанию):")
#
# dataset_path = (str_path if Path(str_path).exists()
#                 else r"C:\Users\IVAN\Desktop\dataEMNIST")


dataset_path = DATASETS[dataset_key].path

# For Future
CRITERIONS = {}

# функция потерь - логарифмическая функция потерь (negative log cross entropy loss)
criterion = nn.NLLLoss()

# задаём остальные параметры
# batch_size = 1000
batch_size_str = input("Введите размер батча(по умолчанию - 1000):")
batch_size = int(batch_size_str) if batch_size_str.isdigit() else 1000

# learning_rate = 0.001
learning_rate_str = input("Скорость обучения(по умолчанию - 10^-3):")
learning_rate = float(batch_size) if learning_rate_str.isdigit() else 0.001

epochs_str = input("Введите число эпох(по умолчанию - 20):")
epochs = int(epochs_str) if epochs_str.isdigit() else 20
print(f"Выбрано {epochs} эпох.")

# (train_loader,test_loader)
data_loader = dataset_loader(batch_size, dataset_path)
train_data, test_data = data_loader.dataset
# a=5
# train_loader
# test_loader
# train_net(net,train[0],optimizer,criterion,device)
# test_nn(net,train[1],device)

# обучение сети
avg_train_acc, train_time_str = execute_with_time_measure(
    train_net,
    net_model, train_data, optimizer,
    criterion, epochs, used_device, epoch_losses,
    acc_list, batch_size)
print(f"Точность при тренировке: {avg_train_acc}")

# тест сети
avg_test_acc, test_time_str = execute_with_time_measure(
    test_net,
    net_model,
    criterion,
    test_data,
    used_device
)
print(f"Точность при тестировании: {avg_test_acc:.3f} %")


# dt_time_stamp = datetime.now()

prepared_name = (f"{DATASETS[dataset_key].name}"
                 f"(op={optimizer_name},ep={epochs},"
                 f"acc={avg_test_acc:.3f})")

print(prepared_name)
save_filepath = "./results" / Path(prepared_name)

# строим график обучения
graphs_shower.graphics_show_loss_acc(epoch_losses, acc_list,
                                     str(save_filepath) + ".png")

time_info = (train_time_str, test_time_str)
# Сохраняем результаты в файл
save_stats(acc_list, epoch_losses, time_info,
           str(save_filepath) + ".txt")

if model_path.exists():
    torch.save(net_model.state_dict(), model_path)
    print("Model Saved!")
