import os
import time
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from numpy import printoptions
import requests
import tarfile
import random
import json
from shutil import copyfile
import pathlib

from typing import List, Dict
from multiprocessing import Pool
from skimage.io import imread
from PIL import Image

import time
import datetime

from tools import get_transform, get_instance_segmentation_model, ObjectDetectionDataSet
from tools import map_class_to_int, save_json, read_json, get_filenames_of_path, adapt_data
from tools import checkpoint_save, show_sample, MutilabelClassificationDataset
from backbones import GetModel
from torchmets import sklearn_metrics

import shutil
import sys

params = {'OWNER': 'rubsini',  # Nombre de usuario en Neptune.ai
          'PROJECT': 'CXR-Classification', # Nombre dle proyecto creado en Neptune.ai
          'SAVE_PATH': 'Checkpoints/', # Donde Guardar los parametros del modelo entrenado
          'LOG_MODEL': True,  # Si se cargará el modelo a neptune después del entrenamiento
          'BACKBONE': 'shufflenet_v2_x0_5', # Red usada para transferencia de aprendizaje
          'BATCH_SIZE': 24, # Tamaño del lote
          'LR': 1e-3, # Tasa de aprendizaje
          'PRECISION': 16, # Precisión de cálculo
          'CLASSES': 9, # Número de clases (incluyendo el fondo)
          'SEED': 42, # Semilla de aleatoreidad
          'EPOCHS': 100, # Número máximo de épocas
          'IMG_MEAN': [0.485, 0.456, 0.406], # Medias de ImageNet (Donde se preentrenaron los modelos)
          'IMG_STD': [0.229, 0.224, 0.225], # Desviaciones estándar de ImageNet (Donde se preentrenaron los modelos)
          'IOU_THRESHOLD': 0.5, # Umbral de Intersección sobre Union para evaluar predicciones en entrenamiento
          'N_WORKERS': 2,
          'WEIGHT_DECAY': 0.0005
          }


def main():
    # Directorio donde se enceuentran las imágenes y etiquetas para entrenamiento
    root = pathlib.Path('/content/drive/MyDrive/ChestXRay8/512')

    # Cargar las imágenes y las etiquetas
    inputs = get_filenames_of_path(root / 'ChestBBImages')
    targets = get_filenames_of_path(root / 'ChestBBLabels')

    # Ordenar entradas y objetivos
    inputs.sort()
    targets.sort()

    # Mapear las etiquetas con valores enteros
    mapping = {'Atelectasis': 1,
               'Cardiomegaly': 3,
               'Effusion': 4,
               'Infiltrate': 8,
               'Mass': 6,
               'Nodule': 7,
               'Pneumonia': 2,
               'Pneumothorax': 5}

    # Participación estratificada: misma cantidad de instancias respecto a sus etiquetas en cada subconjunto
    StratifiedPartition = read_json(pathlib.Path('/content/drive/MyDrive/DatasetSplits/ChestXRay8/split1.json'))

    inputs_train = [pathlib.Path('/content/drive/MyDrive/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Train'].keys())]
    targets_train = [pathlib.Path('/content/drive/MyDrive/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Train'].keys())]

    inputs_valid = [pathlib.Path('/content/drive/MyDrive/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Val'].keys())]
    targets_valid = [pathlib.Path('/content/drive/MyDrive/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Val'].keys())]

    inputs_test = [pathlib.Path('/content/drive/MyDrive/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Test'].keys())]
    targets_test = [pathlib.Path('/content/drive/MyDrive/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Test'].keys())]

    lt = len(inputs_train)+len(inputs_valid)+len(inputs_test)
    ltr,ptr,lvd,pvd,lts,pts = len(inputs_train), len(inputs_train)/lt, len(inputs_valid), len(inputs_valid)/lt, len(inputs_test), len(inputs_test)/lt
    print('Total de datos: {}\nDatos entrenamiento: {} ({:.2f}%)\nDatos validación: {} ({:.2f}%)\nDatos Prueba: {} ({:.2f}%)'.format(lt,ltr,ptr,lvd,pvd,lts,pts))

    if not os.path.exists('ClassifData'):
      os.makedirs('ClassifData')
    adapt_data([inputs_train, targets_train], mapping, "ClassifData/Train_metadata.json")
    adapt_data([inputs_valid, targets_valid], mapping, "ClassifData/Valid_metadata.json")
    adapt_data([inputs_test, targets_test], mapping, "ClassifData/Test_metadata.json")

    # Logging metadata
    import neptune.new as neptune
    from neptune.new.types import File

    # Llave personal de usuario obtenida de Neptune.ai
    NEPTUNE_API_TOKEN = str(sys.argv[1]) #os.getenv("NEPTUNE")
    # Se puede copiar y poner directamente la llave. O configurar como variable de entorno
    run = neptune.init(project=f'{params["OWNER"]}/{params["PROJECT"]}',
                       api_token=NEPTUNE_API_TOKEN)

    run['parameters'] = params
    # run.stop()

    # Initialize the training parameters.
    num_workers = params["N_WORKERS"] #2 # Number of CPU processes for data preprocessing
    lr = params["LR"]# 1e-3 # Learning rate
    batch_size = params["BATCH_SIZE"]# 24
    save_freq = 50 # Save checkpoint frequency (epochs)
    test_freq = 25 # Test model frequency (iterations)
    max_epoch_number = params["EPOCHS"] # 100 # Number of epochs for training
    # Note: on the small subset of data overfitting happens after 30-35 epochs

    device = torch.device('cuda')
    # Save path for checkpoints
    save_path = params['SAVE_PATH']

    # Run tensorboard
    # %load_ext tensorboard
    # %tensorboard --logdir {logdir}

    # Test preprocessing
    val_transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(params["IMG_MEAN"], params["IMG_STD"])
    ])
    print(tuple(np.array(np.array(params["IMG_MEAN"])*255).tolist()))

    # Train preprocessing
    train_transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),
        # transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5),
        #                         shear=None, resample=False,
        #                         fillcolor=tuple(np.array(np.array(mean)*255).astype(int).tolist())),
        transforms.ToTensor(),
        transforms.Normalize(params["IMG_MEAN"], params["IMG_STD"])
    ])

    # Initialize the dataloaders for training.
    test_annotations = os.path.join('ClassifData/Test_metadata.json')
    train_annotations = os.path.join('ClassifData/Train_metadata.json')

    img_folder = '/content/drive/MyDrive/ChestXRay8/512/ChestBBImages'

    test_dataset = MutilabelClassificationDataset(img_folder, test_annotations, val_transform)
    train_dataset = MutilabelClassificationDataset(img_folder, train_annotations, train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    num_train_batches = int(np.ceil(len(train_dataset) / batch_size))

    # Initialize the model
    model = GetModel(len(train_dataset.classes), params['BACKBONE'])
    # Switch model to the training mode and move it to GPU.
    model.train()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=params['WEIGHT_DECAY'])

    # If more than one GPU is available we can use both to speed up the training.
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    os.makedirs(save_path, exist_ok=True)

    # Loss function
    criterion = nn.BCELoss()
    # Tensoboard logger
    # logger = SummaryWriter(logdir)

    # Run training
    epoch = 0
    iteration = 0
    best_macro_F1 = 0.0
    while True:
        batch_losses = []
        for imgs, targets in train_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()

            model_result = model(imgs)
            loss = criterion(model_result, targets.type(torch.float))

            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()

            run["logs/lr"].log(optimizer.param_groups[0]['lr'])
            run["logs/loss_step"].log(loss)

            # logger.add_scalar('train_loss', batch_loss_value, iteration)
            batch_losses.append(batch_loss_value)
            with torch.no_grad():
                result = sklearn_metrics(model_result.cpu().numpy(), targets.cpu().numpy())
                for metric in result:
                    # logger.add_scalar('train/' + metric, result[metric], iteration)
                    run["logs/train_"+metric.replace('/','_')].log(result[metric])

            if iteration % test_freq == 0:
                model.eval()
                with torch.no_grad():
                    model_result = []
                    targets = []
                    for imgs, batch_targets in test_dataloader:
                        imgs = imgs.to(device)
                        model_batch_result = model(imgs)
                        model_result.extend(model_batch_result.cpu().numpy())
                        targets.extend(batch_targets.cpu().numpy())

                result = sklearn_metrics(np.array(model_result), np.array(targets))
                res_macro_F1 = result['macro/f1']
                for metric in result:
                    # logger.add_scalar('test/' + metric, result[metric], iteration)
                    run["logs/test_"+metric.replace('/','_')].log(result[metric])
                print("epoch:{:2d} iter:{:3d} test: "
                      "micro f1: {:.3f} "
                      "macro f1: {:.3f} "
                      "samples f1: {:.3f}".format(epoch, iteration,
                                                  result['micro/f1'],
                                                  result['macro/f1'],
                                                  result['samples/f1']))

                model.train()
            iteration += 1

        loss_value = np.mean(batch_losses)
        run["logs/loss_epoch"].log(loss_value)
        print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
        if best_macro_F1 < res_macro_F1 or epoch % save_freq == 0:
            if best_macro_F1 <= res_macro_F1:
                best_macro_F1 = res_macro_F1
                s_model, s_epoch = model, epoch
        if epoch % save_freq == 0:
            checkpoint_save(s_model, save_path, s_epoch, run)
        epoch += 1
        if max_epoch_number < epoch:
            break

    run.stop()
    shutil.rmtree("Checkpoints")

if __name__ == "__main__":
    main()
