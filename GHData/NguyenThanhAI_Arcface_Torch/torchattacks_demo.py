import os
import argparse
from typing import List
from tqdm import tqdm

import numpy as np

import cv2

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchvision import transforms

import torchattacks
#from torchattacks import PGD, APGD, APGDT, AutoAttack, RFGSM, FFGSM
from torchattacks.attack import Attack
from torchattacks import VANILA, GN, FGSM, BIM, CW, RFGSM, PGD, PGDL2, EOTPGD, TPGD, FFGSM, \
    MIFGSM, APGD, APGDT, FAB, Square, AutoAttack, OnePixel, DeepFool, SparseFool, DIFGSM, UPGD, TIFGSM, Jitter, \
    Pixle

from backbones import get_model
from utils_fn import enumerate_images

from facenet_pytorch import InceptionResnetV1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="r50")
    parser.add_argument("--weights", type=str, default=r"C:\Users\Thanh\Downloads\backbone.pth")
    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\\aligned_faces")
    #parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\facenet_pytorch_torchattacks_images")
    parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\torchattacks_images")

    args = parser.parse_args()

    return args


class FaceDataset(Dataset):
    def __init__(self, images_dir: str) -> None:
        super().__init__()
        self.images_list = enumerate_images(images_dir=images_dir)
        identities = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], self.images_list))))
        identities.sort()
        identities_to_id = dict(zip(identities, list(range(1, len(identities) + 1))))
        self.images_to_groundtruth_id = dict(zip(self.images_list, list(map(lambda x: identities_to_id[os.path.normpath(x).split(os.sep)[-2]], self.images_list))))
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.images_list[index]
        label = self.images_to_groundtruth_id[image]

        img = cv2.imread(image)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        #img.div_(255).sub_(0.5).div_(0.5)
        img.div_(255)

        return img, label


if __name__ == "__main__":

    args = get_args()

    network = args.network
    weights = args.weights
    images_dir = args.images_dir
    save_dir = args.save_dir

    #attack_types: List[Attack] = [VANILA, GN, FGSM, BIM, CW, RFGSM, PGD, PGDL2, EOTPGD, TPGD, FFGSM, \
    #MIFGSM, APGD, APGDT, FAB, Square, AutoAttack, OnePixel, DeepFool, SparseFool, DIFGSM, UPGD, TIFGSM, Jitter, \
    #Pixle]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    images_list = enumerate_images(images_dir=images_dir)
    identities = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], images_list))))
    identities.sort()
    identities_to_id = dict(zip(identities, list(range(1, len(identities) + 1))))
    id_to_identities = {v: k for k, v in identities_to_id.items()}
    images_to_groundtruth_id = dict(zip(images_list, list(map(lambda x: identities_to_id[os.path.normpath(x).split(os.sep)[-2]], images_list))))
    image_ord_to_image_name = dict(zip(range(len(images_list)), images_list))


    model = get_model(name=network)
    model.load_state_dict(torch.load(weights))
    #model = InceptionResnetV1(pretrained="casia-webface")
    model.eval()
    model.to(device)

    attack_types: List[Attack] = [
    FGSM(model, eps=8/255),
    BIM(model, eps=8/255, alpha=2/255, steps=100),
    RFGSM(model, eps=8/255, alpha=2/255, steps=100),
    CW(model, c=1, lr=0.01, steps=100, kappa=0),
    PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True),
    PGDL2(model, eps=1, alpha=0.2, steps=100),
    EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    FFGSM(model, eps=8/255, alpha=10/255),
    TPGD(model, eps=8/255, alpha=2/255, steps=100),
    MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1),
    VANILA(model),
    GN(model, std=0.1),
    APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
    FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    AutoAttack(model, eps=8/255, n_classes=10, version='standard'),
    OnePixel(model, pixels=5, inf_batch=50),
    DeepFool(model, steps=100),
    DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
    SparseFool(model, steps=100, lam=10, overshoot=0.05),
    UPGD(model, eps=8/255, alpha=4/255, steps=100, loss="dlr"),
    TIFGSM(model, eps=8/255, alpha=4/255, steps=100),
    Jitter(model, eps=0.4, alpha=4/255, steps=100, std=0.1),
    Pixle(model, update_each_iteration=True)
    ]

    dataset = FaceDataset(images_dir=images_dir)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    for images, labels in dataloader:
        #print(images.numpy().shape, labels.numpy().shape)
        print(images.min(), images.max())

    
    #for attack_type in tqdm(attack_types):
    for atk in tqdm(attack_types):

        #atk = PGD(model, eps=32/255, alpha=4/255, steps=100)
        #atk = APGD(model=model, eps=16/255, verbose=True)
        #atk = APGDT(model=model)
        #atk = RFGSM(model, steps=10)
        #atk = attack_type(model)
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

        #atk = APGD()

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
