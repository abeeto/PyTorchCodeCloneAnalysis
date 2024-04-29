import argparse
import datetime
import json
import os
import pickle
import platform
import time

import pandas as pd
import psutil
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import glob
import numpy as np


# SETUP
torch.backends.cudnn.benchmark = True

# Training settings
# precisions = ["float", "half", "double"]
precisions = ["float"]
parser = argparse.ArgumentParser(description="NGD PyTorch Benchmarking")
parser.add_argument("--WARM-UP", "-w", type=int, default=5, required=False, help="Number of warm-up rounds")
parser.add_argument("--SAVED_DATA_FOLDER", "-d", type=str, default="./sample-data", required=False, help="Folder where data is saved")
parser.add_argument("--BATCH_SIZE", "-b", type=int, default=4, required=False, help="Batch size")
parser.add_argument("--NUM_CLASSES", "-c", type=int, default=100, required=False, help="Number of classes")
parser.add_argument("--RESULT_SAVE_FOLDER", "-r", type=str, default="./results", required=False, help="folder to save results")
args = parser.parse_args()

# List of models we want to test
MODEL_LIST = {
    # models.mnasnet: models.mnasnet.__all__[1:],
    # models.resnet: models.resnet.__all__[1:],
    # models.densenet: models.densenet.__all__[1:],
    # models.squeezenet: models.squeezenet.__all__[1:],
    models.vgg: models.vgg.__all__[1:],
    models.mobilenet: models.mobilenet.mv2_all[1:],
    # models.mobilenet: models.mobilenet.mv3_all[1:],
    # models.shufflenetv2: models.shufflenetv2.__all__[1:],
}


# LOAD NPY DATA FILES
class LoadSavedDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, index):
        data = np.load(self.file_list[index])
        return self.transform(data.transpose(1, 2, 0))

    def __len__(self):
        return len(self.file_list)


file_list = glob.glob(os.path.join(args.SAVED_DATA_FOLDER, '*.npy'))
dataset = LoadSavedDataset(file_list, transform=transforms.Compose([transforms.ToTensor()]))
data_loader = DataLoader(
    dataset=dataset,
    batch_size=args.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


def train(device, precision="single", num_gpus=1):
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():

        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            # I case we want to do multi-gpu test
            if num_gpus > 1:
                model = nn.DataParallel(model, device_ids=range(num_gpus))
            model = getattr(model, precision)()
            model = model.to(device)
            durations = []
            print("Benchmarking Training for model:", model_name, "With precision type:", precision)
            total_duration_start = time.time()
            for step, img in enumerate(data_loader):
                img = getattr(img, precision)()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model(img.to(device))
                loss = criterion(prediction, target)
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                # Collect time after the warm up rounds
                if step >= args.WARM_UP:
                    durations.append((end - start) * 1000)
            total_duration_end = time.time()
            print("\tAvarage Training Time:", (sum(durations) / len(durations)), "ms")
            print("\tTotal Duration:", ((total_duration_end - total_duration_start) * 1000), "ms")
            del model

            benchmark[len(benchmark) + 1] = {"model_type": model_type.__name__, "model_name": model_name,
                                             "durations": durations,
                                             "total_duration": (total_duration_end - total_duration_start) * 1000}
    return benchmark


def inference(device, precision="float", num_gpus=1):
    benchmark = {}
    with torch.no_grad():
        for model_type in MODEL_LIST.keys():

            for model_name in MODEL_LIST[model_type]:
                model = getattr(model_type, model_name)(pretrained=False)
                if num_gpus > 1:
                    model = nn.DataParallel(model, device_ids=range(num_gpus))
                model = getattr(model, precision)()
                model = model.to(device)
                model.eval()
                durations = []
                print("Benchmarking Inference for model:", model_name, "With precision type:", precision)
                total_duration_start = time.time()
                for step, img in enumerate(data_loader):
                    img = getattr(img, precision)()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.time()
                    model(img.to(device))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start) * 1000)
                total_duration_end = time.time()
                print("\tAvarage Inference Time:", (sum(durations) / len(durations)), "ms")
                print("\tTotal Duration:", ((total_duration_end - total_duration_start) * 1000), "ms")
                del model
                benchmark[len(benchmark) + 1] = {"model_type": model_type.__name__, "model_name": model_name,
                                                 "durations": durations,
                                                 "total_duration": (total_duration_end - total_duration_start) * 1000}
    return benchmark


if __name__ == "__main__":
    config = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config["Device Info"] = \
        {
            "Using device": str(device)
        }

    # Print some system info
    config["System Info"] = \
        {
            "Platform Name": platform.uname(),
            "CPU Frequency": psutil.cpu_freq(),
            "CPU Count": psutil.cpu_count(),
            "Memory Available": str(psutil.virtual_memory().total / (1024.0 ** 3)) + "GB",
        }

    # When cuda is available, print more info
    if device.type == 'cuda':
        config["GPU Info"] = \
            {
                "Number of GPUs on current device": torch.cuda.device_count(),
                "Allocated": str(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)) + "GB",
                "Cached": str(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)) + "GB",
                "CUDA Version": torch.version.cuda,
                "Cudnn Version": torch.backends.cudnn.version()
            }

    config["Config"] = vars(args)
    print(json.dumps(config, sort_keys=True, indent=4))

    # Start the timer
    now = datetime.datetime.now()
    start_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print("\n\nBenchmark Started:", start_time)
    print("-" * 50)

    # TRAINING/INFERENCE GOES HERE
    for precision in precisions:
        # TRAINING
        train_result = train(device=device, precision=precision)
        train_result_df = pd.DataFrame()
        for k, v in train_result.items():
            train_result_df = train_result_df.append(v, ignore_index=True)
        print("-" * 50)
        # INFERENCE
        inference_result = inference(device=device, precision=precision)
        inference_result_df = pd.DataFrame()
        for k, v in inference_result.items():
            inference_result_df = inference_result_df.append(v, ignore_index=True)

    # TRAINING/INFERENCE GOES HERE
    now = datetime.datetime.now()
    end_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print("-" * 50)
    print("Benchmark ENDED:", end_time)

    # Save CONFIG
    run_id = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
    with open(os.path.join(args.RESULT_SAVE_FOLDER, run_id + '_config.pkl'), 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

    # Save TRAINING Results
    train_result_df.to_csv(os.path.join(args.RESULT_SAVE_FOLDER, run_id + '_training.csv'), index=True)

    # Save INFERENCE Results
    inference_result_df.to_csv(os.path.join(args.RESULT_SAVE_FOLDER, run_id + '_inference.csv'), index=True)
