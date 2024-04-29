import os
import argparse
from typing import List
from tqdm import tqdm

import numpy as np

import cv2

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn


import torchattacks

from torchattacks.attack import Attack
from torchattacks import MultiAttack, VANILA, GN, FGSM, BIM, CW, RFGSM, PGD, PGDL2, EOTPGD, TPGD, FFGSM, \
    MIFGSM, APGD, APGDT, FAB, Square, AutoAttack, OnePixel, DeepFool, SparseFool, DIFGSM, UPGD, TIFGSM, Jitter, \
    Pixle

import foolbox as fb

import art.attacks.evasion as evasion
from art.estimators.classification import PyTorchClassifier

from backbones import get_model
from utils_fn import enumerate_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default=r"D:\Face_Datasets\CelebA_Models\checkpoint.pth")
    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    #parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\facenet_pytorch_torchattacks_images")
    parser.add_argument("--save_dir", type=str, default=r"D:\Face_Datasets\choose_train_torchattacks_images")

    args = parser.parse_args()

    return args


'''class FaceDataset(Dataset):
    def __init__(self, images_dir: str) -> None:
        super().__init__()
        self.images_list = enumerate_images(images_dir=images_dir)
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.images_list[index]
        label = int(image.split(os.sep)[-2])

        img = cv2.imread(image)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        img.div_(255)

        return img, label'''



class L2BrendelBethge(Attack):
    def __init__(self, model):
        super(L2BrendelBethge, self).__init__("L2BrendelBethge", model)
        self.fmodel = fb.PyTorchModel(self.model, bounds=(0,1), device=self.device)
        self.init_attack = fb.attacks.DatasetAttack()
        self.adversary = fb.attacks.L2BrendelBethgeAttack(init_attack=self.init_attack)
        self._attack_mode = 'only_default'
        
    def forward(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)
        
        # DatasetAttack
        batch_size = len(images)
        batches = [(images[:batch_size//2], labels[:batch_size//2]),
                   (images[batch_size//2:], labels[batch_size//2:])]
        self.init_attack.feed(model=self.fmodel, inputs=batches[0][0]) # feed 1st batch of inputs
        self.init_attack.feed(model=self.fmodel, inputs=batches[1][0]) # feed 2nd batch of inputs
        criterion = fb.Misclassification(labels)
        init_advs = self.init_attack.run(self.fmodel, images, criterion)
        
        # L2BrendelBethge
        adv_images = self.adversary.run(self.fmodel, images, labels, starting_points=init_advs)
        return adv_images


class JSMA(Attack):
    def __init__(self, model, theta=1/255, gamma=0.15, batch_size=128):
        super(JSMA, self).__init__("JSMA", model)
        self.classifier = PyTorchClassifier(
                            model=self.model, clip_values=(0, 1),
                            loss=nn.CrossEntropyLoss(),
                            optimizer=optim.Adam(self.model.parameters(), lr=0.01),
                            input_shape=(1, 28, 28), nb_classes=10)
        self.adversary = evasion.SaliencyMapMethod(classifier=self.classifier,
                                                   theta=theta, gamma=gamma,
                                                   batch_size=batch_size)
        self.target_map_function = lambda labels: (labels+1)%10
        self._attack_mode = 'only_default'
        
    def forward(self, images, labels):
        adv_images = self.adversary.generate(images, self.target_map_function(labels))
        return torch.tensor(adv_images).to(self.device)


class FaceDataset(Dataset):
    def __init__(self, images_dir: str) -> None:
        super().__init__()
        self.images_list = enumerate_images(images_dir=images_dir)
        identities = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], self.images_list))))
        identities.sort()
        identities_to_id = dict(zip(identities, list(range(len(identities)))))
        print(identities_to_id)
        self.images_to_groundtruth_id = dict(zip(self.images_list, list(map(lambda x: identities_to_id[os.path.normpath(x).split(os.sep)[-2]], self.images_list))))
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.images_list[index]
        label = int(self.images_to_groundtruth_id[image])
        #print(image, label)

        img = cv2.imread(image)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        #img.div_(255).sub_(0.5).div_(0.5)
        img.div_(255)

        return img, label


class FaceModel(nn.Module):

    def __init__(self, model_name: str="r18", num_classes: int=10177):
        super().__init__()
        self.backbone = get_model(name=model_name)

        #for layer in self.backbone.parameters():
        #    layer.requires_grad = False
        

        #in_features = self.backbone.features.out_features
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, images):
        x = self.backbone(images)
        output = self.fc(x)

        return output


if __name__ == "__main__":
    args = get_args()

    weights = args.weights
    images_dir = args.images_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    '''images_list = enumerate_images(images_dir=images_dir)
    images_list.sort(key=lambda x: int(os.path.normpath(x).split(os.sep)[-2]))
    image_ord_to_image_name = dict(zip(range(len(images_list)), images_list))'''

    images_list = enumerate_images(images_dir=images_dir)
    identities = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], images_list))))
    identities.sort()
    identities_to_id = dict(zip(identities, list(range(len(identities)))))
    id_to_identities = {v: k for k, v in identities_to_id.items()}
    images_to_groundtruth_id = dict(zip(images_list, list(map(lambda x: identities_to_id[os.path.normpath(x).split(os.sep)[-2]], images_list))))
    image_ord_to_image_name = dict(zip(range(len(images_list)), images_list))

    #model = torchvision.models.resnet18(num_classes=10177)
    model = FaceModel(model_name="r50", num_classes=len(identities))
    model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))["weights"])

    model.eval()
    model.to(device=device)

    attack_types: List[Attack] = [
    #FGSM(model, eps=32/255),
    #PGD(model, eps=0.6, steps=50, random_start=True),
    #CW(model, kappa=0.5),
    #L2BrendelBethge(model),
    AutoAttack(model, n_classes=len(identities)),
    #JSMA(model, theta=8/255, gamma=0.3),
    #PGDL2(model, eps=2, alpha=0.4),
    #EOTPGD(model, eps=0.4, alpha=8/255)
    ]

    dataset = FaceDataset(images_dir=images_dir)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    #for images, labels in dataloader:
    #    print(images.min(), images.max())

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