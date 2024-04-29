import os
import argparse
import yaml
import torch
from torch import nn
from tqdm import tqdm
from _collections import defaultdict

from utils.dataset import DatasetBuilder
from utils.utils import get_device, get_logger
from models.common import load_model


def get_classes(classes_path):
    with open(classes_path, 'r') as f:
        classes = f.read().split('\n')
    classes = [cl for cl in classes if cl]
    return classes


def evaluate(model, dataset, device, classes=None):
    """ Get predictions from model and calculate accuracy
    Args:
        model: PyTorch Model
        data: List of images
        device: cuda or cpu
    Return:

    """
    if classes:
        classes_accuracy = {}
        for cl_name in classes:
            classes_accuracy[cl_name] = {"correct": 0, "examples": 0}
    correct = 0
    for num in tqdm(range(len(dataset))):
        img, label = dataset[num]
        img = img.unsqueeze(0).cuda(device)
        pred = model(img)
        pred = nn.Softmax()(pred)
        predicted = int(torch.max(pred.data, 1)[1])
        # print(label)
        # print(predicted)
        if predicted == label:
            correct += 1
        if classes:
            classes_accuracy[classes[int(label)]]["examples"] += 1
            if predicted == label:
                classes_accuracy[classes[int(label)]]["correct"] += 1
        # if num > 5:
        #     break
    acc = int(correct / len(dataset) * 100)
    print(f"Total Accuracy: {acc} %")
    if classes:
        print("Accuracy by classes")
        for cl_name, val in classes_accuracy.items():
            acc = int(val["correct"] / val["examples"] * 100)
            print(f"{cl_name}: {acc}%")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='',
                        help="Path to model")
    parser.add_argument("--dataset", type=str, default='',
                        help="Path to dataset")
    parser.add_argument("--config", type=str, default='',
                        help="Path to config file")
    parser.add_argument("--device", type=str, default='0',
                        help="Device: 'cpu', '0', '1', ...")
    parser.add_argument("--accuracy-by-class", action="store_true",
                        help="Display accuracy for each class")
    args = parser.parse_args()

    if os.path.isfile(args.config):
        with open(args.config, 'r') as cfg_file:
            config = yaml.load(cfg_file, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Path {args.config} not found")

    logger = get_logger()
    device = get_device(args.device)
    model = load_model(arch=config["Train"]["arch"], num_classes=config["Dataset"]["num_classes"],
                       device=device, model_path=args.model, channels=config["Train"]["channels"],
                       logger=logger)

    if os.path.isfile(args.dataset):
        dataset = DatasetBuilder(args.dataset, config["Dataset"]["classes_path"], config["Train"]["image_size"], logger)
    else:
        raise ValueError(f"Path {args.dataset} not found")

    if args.accuracy_by_class:
        classes = get_classes(config["Dataset"]["classes_path"])
    else:
        classes = None

    evaluate(model, dataset, device, classes)
