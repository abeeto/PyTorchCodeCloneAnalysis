import argparse
import os

import math

import random

from itertools import groupby

from PIL import Image
from requests import get

from tqdm import tqdm

import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from backbones import get_model

from utils_fn import enumerate_images


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster
    print(f'Setting all seeds to be {seed} to reproduce...')

seed_everything(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceDataset(Dataset):
    def __init__(self, images_dir: str, subset: str="train") -> None:
        super().__init__()
        self.images_list = enumerate_images(images_dir=images_dir)
        self.class_list = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], self.images_list))))
        self.class_list.sort()
        self.class_to_label = dict(zip(self.class_list, range(len(self.class_list))))
        print(self.class_to_label)
        #print(self.class_to_label)
        self.transfrom = transforms.Compose([transforms.Resize([112, 112]), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.images_list[index]
        label = int(self.class_to_label[os.path.normpath(image).split(os.sep)[-2]])

        #img = torchvision.io.read_image(image)
        img = Image.open(image).convert("RGB")
        
        #img.div_(255).sub_(0.5).div_(0.5)
        img = self.transfrom(img)

        return img, label


class FaceModel(nn.Module):

    def __init__(self, model_name: str="r18", num_classes: int=10177):
        super().__init__()
        self.backbone = get_model(name=model_name)

        for layer in self.backbone.parameters():
            layer.requires_grad = False
        

        #in_features = self.backbone.features.out_features
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, images):
        x = self.backbone(images)
        output = self.fc(x)

        return output



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    parser.add_argument("--val_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    parser.add_argument("--model_dir", type=str, default=r"D:\Face_Datasets\CelebA_Models")
    parser.add_argument("--checkpoint_pattern", type=str, default=r"checkpoint")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    args = parser.parse_args()

    return args


def train_epoch(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: optim.Optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred = model(images)
        loss = loss_fn(pred, labels)

        loss.backward()

        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(images)
            print("Loss: {}, [{}/{}]".format(loss, current, size))


def val_loop(dataloader: DataLoader, model: nn.Module, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)

            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
    test_loss /= num_batch
    correct /= size
    print("Accuracy: {}%, Avg loss: {}".format(correct * 100, test_loss))
    return correct, test_loss


def save_model(model: nn.Module, accuracy: float, loss: float, epoch: int, save_path: str):
    torch.save({"weights": model.state_dict(),
                "accuracy": accuracy,
                "loss": loss,
                "epoch": epoch}, save_path)
    print("Save model with accuracy: {}, loss {} at epoch: {}".format(accuracy, loss, epoch))


if __name__ == "__main__":

    args = get_args()

    train_dir = args.train_dir
    val_dir = args.val_dir
    model_dir = args.model_dir
    checkpoint_pattern = args.checkpoint_pattern
    pretrained = args.pretrained
    num_epochs  = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    train_dataset = FaceDataset(images_dir=train_dir)
    val_dataset = FaceDataset(images_dir=val_dir)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = FaceModel(model_name="r50", num_classes=len(train_dataset.class_list))

    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location=torch.device("cpu"))["weights"])
    model.to(device=device)
    print("Model: {}".format(model))


    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    max_accuracy = -np.inf
    save_path = os.path.join(model_dir, checkpoint_pattern + ".pth")
    for t in range(num_epochs):
        print("Epoch {}\n-------------------------------------------------".format(t + 1))
        train_epoch(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        val_acc, val_loss = val_loop(dataloader=val_dataloader, model=model, loss_fn=loss_fn)
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            save_model(model=model, accuracy=val_acc, loss=val_loss, epoch=t+1, save_path=save_path)
    print("Done")
    