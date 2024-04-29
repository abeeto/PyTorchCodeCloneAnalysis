import torch
from covid import mean, std, label_statistics
from covid_model import load_model, loss_epoch
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.datasets import ImageFolder

from covid_model import test, plot_confusion_matrix
from covid import get_data_loader
import os
import pathlib, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prameter for ML')
    parser.add_argument('-O', type=pathlib.Path, help='Output folder path', required=True)
    parser.add_argument('-I', type=pathlib.Path, help='input image folder path', required=True)
    parser.add_argument('-M', choices=['resnet', 'vgg'], required=True)
    parser.add_argument('-P', choices=['T', 'F'], required=True)
    parser.add_argument('-p', choices=['T', 'F'], required=True)
    parser.add_argument('-E', type=int, required=True)
    parser.add_argument('-B', type=int, required=True)
    parser.add_argument('-m', type=int, required=True)
    parser.add_argument('-g', type=int, required=True)
    
    parsed = parser.parse_args()

    output_folder = str(parsed.O)
    input_data = str(parsed.I)
    model_name = parsed.M
    pretrained = parsed.P == 'T'
    fc_only = parsed.p == 'T'
    num_epochs = parsed.E
    batch_size = parsed.B
    max_data = parsed.m
    gpu_id = parsed.g
    
    try:
        os.makedirs(output_folder, exist_ok=False)
    except FileExistsError:
        print("Overwriting output folder!")

    device = torch.device("cuda:%d" % gpu_id)
    # Load model and data
    model = load_model(model_name, pretrained, fc_only, device)
    model.load_state_dict(torch.load(os.path.join(output_folder, "weight.pt")))

    train_dl, val_dl, test_dl = get_data_loader(input_data, batch_size, output_folder, max_data)
    

    print("############### Test Phase ###############")
    cf = test(model, test_dl, device, output_folder)
    plot_confusion_matrix(cf, ["non Covid19", "Covid19"], output_folder)
