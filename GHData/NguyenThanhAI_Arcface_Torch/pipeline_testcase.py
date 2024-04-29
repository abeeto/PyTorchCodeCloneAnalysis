import os
import argparse
from typing import List
from tqdm import tqdm

import random

import shutil

from PIL import Image

import numpy as np

import cv2

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import torch.nn.functional as F

import torch.nn as nn


import torchattacks

from torchattacks.attack import Attack
from torchattacks import MultiAttack, VANILA, GN, FGSM, BIM, CW, RFGSM, PGD, PGDL2, EOTPGD, TPGD, FFGSM, \
    MIFGSM, APGD, APGDT, FAB, Square, AutoAttack, OnePixel, DeepFool, SparseFool, DIFGSM, UPGD, TIFGSM, Jitter, \
    Pixle

import imgaug as ia
from imgaug import augmenters as iaa

from backbones import get_model
from utils_fn import enumerate_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def imgaug_function() -> iaa.Sequential:
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)


    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Sometimes(0.3, iaa.Cutout(nb_iterations=(1, 3), size=(0.05, 0.1), squared=False)),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    return seq


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default=r"D:\Face_Datasets\CelebA_Models\checkpoint.pth")
    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    #parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\facenet_pytorch_torchattacks_images")
    parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\choose_train_torchattacks_images")

    parser.add_argument("--train_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    parser.add_argument("--val_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    parser.add_argument("--model_dir", type=str, default=r"D:\Face_Datasets\CelebA_Models")
    parser.add_argument("--checkpoint_pattern", type=str, default=r"checkpoint")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--face_attributes_edit_dir", type=str, default=r"D:\Face_Datasets\aligned_choose_train_transformed_faces")

    args = parser.parse_args()

    return args


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

    weights = args.weights
    images_dir = args.images_dir
    save_dir = args.save_dir

    train_dir = args.train_dir
    val_dir = args.val_dir
    model_dir = args.model_dir
    checkpoint_pattern = args.checkpoint_pattern
    pretrained = args.pretrained
    num_epochs  = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    face_attributes_edit_dir = args.face_attributes_edit_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

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

    print("Training holdout model")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    #max_accuracy = -np.inf
    #save_path = os.path.join(model_dir, checkpoint_pattern + ".pth")
    for t in range(num_epochs):
        print("Epoch {}\n-------------------------------------------------".format(t + 1))
        train_epoch(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        val_acc, val_loss = val_loop(dataloader=val_dataloader, model=model, loss_fn=loss_fn)
        #if val_acc > max_accuracy:
        #    max_accuracy = val_acc
        #    save_model(model=model, accuracy=val_acc, loss=val_loss, epoch=t+1, save_path=save_path)
    print("Done training holdout model")
    

    images_list = enumerate_images(images_dir=images_dir)
    identities = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], images_list))))
    identities.sort()
    identities_to_id = dict(zip(identities, list(range(len(identities)))))
    id_to_identities = {v: k for k, v in identities_to_id.items()}
    images_to_groundtruth_id = dict(zip(images_list, list(map(lambda x: identities_to_id[os.path.normpath(x).split(os.sep)[-2]], images_list))))
    image_ord_to_image_name = dict(zip(range(len(images_list)), images_list))

    model.eval()

    attack_types: List[Attack] = [
    FGSM(model, eps=32/255),
    PGD(model, eps=0.6, steps=50, random_start=True),
    CW(model, kappa=0.5),
    #L2BrendelBethge(model),
    AutoAttack(model, n_classes=len(identities)),
    #JSMA(model, theta=8/255, gamma=0.3),
    #PGDL2(model, eps=2, alpha=0.4),
    #EOTPGD(model, eps=0.4, alpha=8/255)
    ]

    dataset = FaceDataset(images_dir=images_dir)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    print("Start creating adversarial test cases")

    for atk in tqdm(attack_types):
        #atk.set_mode_targeted_least_likely(5)
        atk.set_return_type("int")

        atk_name = atk.__class__.__name__
        print("Attack name: {}".format(atk_name))
        atk_save_dir = os.path.join(save_dir, atk_name)
        if not os.path.exists(atk_save_dir):
            os.makedirs(atk_save_dir, exist_ok=True)

        try:
            atk.save(data_loader=dataloader, save_path=os.path.join(atk_save_dir, "adv_data.pt"), verbose=True)
        except Exception as e:
            print("Error: {}".format(e))
            continue

        adv_images, adv_labels = torch.load(os.path.join(atk_save_dir, "adv_data.pt"))

        print(adv_images.min(), adv_images.max())

        adv_dataset = TensorDataset(adv_images)

        for i, adv_image in enumerate(adv_dataset):
            img = adv_image[0].numpy()
            img = np.transpose(img, (1, 2, 0))
            #print(img.shape)
            image_path = image_ord_to_image_name[i]
            label_id = images_to_groundtruth_id[image_path]
            label = id_to_identities[label_id]
            #print(label)
            img_save_dir = os.path.join(atk_save_dir, label)
            img_save_name = os.path.basename(image_path)
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir, exist_ok=True)
            img_save_path = os.path.join(img_save_dir, img_save_name)
            cv2.imwrite(img_save_path, img[:, :, ::-1])
    
    print("Done creating adversarial test case")
    
    seq = imgaug_function()

    print("Start creating augmentation test case")

    atk_save_dir = os.path.join(save_dir, "imgaug")
    if not os.path.exists(atk_save_dir):
        os.makedirs(atk_save_dir, exist_ok=True)
    for image in tqdm(images_list):
        img = cv2.imread(image)
        img = img[:, :, ::-1]
        images = np.array([img for _ in range(4)], dtype=np.uint8)
        aug_images = seq(images=images)
        rel_path = os.path.join(*image.split(os.sep)[-2:])
        rel_dir = os.path.dirname(rel_path)
        img_save_dir = os.path.join(atk_save_dir, rel_dir)
        basename = os.path.basename(image)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir, exist_ok=True)

        for i, aug_img in enumerate(aug_images):
            savename = os.path.splitext(basename)[0] + "_" + str(i) + os.path.splitext(basename)[1]
            res_path = os.path.join(img_save_dir, savename)
            cv2.imwrite(res_path, aug_img[:, :, ::-1])

    print("Done creating augmentation test case")
    
    face_attribute_images_list = enumerate_images(face_attributes_edit_dir)

    face_attribute_identities = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], images_list))))
    face_attribute_identities.sort()

    if face_attribute_identities == identities:
        print("Same identity, copy face attribute folder")
        copytree(face_attributes_edit_dir, save_dir)
    else:
        print("Not same identity")
    
    print("Done copy")