from typing import List, Dict
import pathlib
from multiprocessing import Pool
from skimage.io import imread

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

import time
import datetime
import json

import utils
import transforms as T
from matplotlib import pyplot as plt
from numpy import printoptions

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """
    Regresa una lista de archivos en un directorio, dado como objeto de pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames

def read_json(path: pathlib.Path):
    with open(str(path), "r") as fp:
        file = json.loads(s=fp.read())
        fp.close()
    return file

def save_json(obj, path: pathlib.Path):
    with open(path, "w") as fp:
        json.dump(obj=obj, fp=fp, indent=4, sort_keys=False)

def map_class_to_int(labels: List[str], mapping: dict):
    """Mapea una cadena (string) a un entero (int)."""
    labels = np.array(labels)
    dummy = np.empty_like(labels)
    for key, value in mapping.items():
        dummy[labels == key] = value

    return dummy.astype(np.uint8)

class ObjectDetectionDataSet(torch.utils.data.Dataset):
    """
    Construye un conjunto de datos con imágenes y sus respectivas etiquetas (objetivos).
    Cada target es esperado que se encuentre en un archivo JSON individual y debe contener
    al menos las llaves 'boxes' y 'labels'.
    Las entradas (imágenes) y objetivos (etiquetas) son esperadas como una lista de
    objetos pathlib.Path

    En caso de que las etiquetas esten en formato string, puedes usar un diccionario de
    mapeo para codificarlas como enteros (int).

    Regresa un diccionario con las siguientes llaves: 'x', 'y'->('boxes','labels'), 'x_name', 'y_name'
    """

    def __init__(
        self,
        inputs: List[pathlib.Path],
        targets: List[pathlib.Path],
        transform = None,
        add_dim: bool = False,
        use_cache: bool = False,
        convert_to_format: str = None,
        mapping: Dict = None,
        tgt_int64: bool = False,
        metadata_dir: pathlib.Path = None,
        filters: List = None,
        id_column: str = None
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.add_dim = add_dim
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping
        self.tgt_int64 = tgt_int64
        self.metadata = metadata_dir
        self.filters = filters
        self.id_column = id_column

        if self.use_cache:
            # Usar multiprocesamiento para cargar las imagenes y las etiquetas en la memoria RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets))

        if metadata_dir:
            self.filtered_inputs = []
            self.filtered_targets = []
            self.id_list = self.add_filters(self.metadata, self.filters, self.id_column)
            for num,input in enumerate(self.inputs):
                if re.search(r'.*\\(.*)\..*', str(input)).group(1) in self.id_list:
                    self.filtered_inputs.append(input)
                    self.filtered_targets.append(self.targets[num])
            self.inputs = self.filtered_inputs
            self.targets = self.filtered_targets


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Seleccionar una muestra
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Cargar entradas (imágenes) y objetivos (etiquetas)
            x, y = self.read_images(input_ID, target_ID)

        # # De RGBA a RGB
        # if x.shape[-1] == 4:
        #     x = rgba2rgb(x)

        # Leer cajas
        try:
            boxes = torch.from_numpy(y["boxes"]).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y["boxes"]).to(torch.float32)

        # Leer puntajes
        if "scores" in y.keys():
            try:
                scores = torch.from_numpy(y["scores"]).to(torch.float32)
            except TypeError:
                scores = torch.tensor(y["scores"]).to(torch.float32)

        # Mapeo de etiquetas
        if self.mapping:
            labels = map_class_to_int(y["labels"], mapping=self.mapping)
        else:
            labels = y["labels"]

        # Leer etiquetas
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)

        # Convertir formato
        if self.convert_to_format == "xyxy":
            boxes = box_convert(
                boxes, in_fmt="xywh", out_fmt="xyxy"
            )  # Transformaciones de las cajas del formato xywh a xyxy
        elif self.convert_to_format == "xywh":
            boxes = box_convert(
                boxes, in_fmt="xyxy", out_fmt="xywh"
            )  # # Transformaciones de las cajas del formato xyxy a xywh

        # Crear objetivos
        tgt = {"boxes": boxes, "labels": labels}

        if "scores" in y.keys():
            target["scores"] = scores

        # Preprocesamiento
        tgt = {
            key: value.numpy() for key, value in tgt.items()
        }  # Todos los tensores debieren ser convertidos a np.ndarrays

        if self.transform is not None:
            x, tgt = self.transform(x, tgt)  # Regresa np.ndarrays

        if "scores" in y.keys():
            bxs,lbs,srs = [],[],[]
            for r,f in enumerate(tgt['scores']):
                if f > 0.70:
                    bxs.append(tgt['boxes'][r])
                    lbs.append(tgt['labels'][r])
                    srs.append(tgt['scores'][r])
            tgt = {'boxes':np.array(bxs), 'labels':np.array(lbs), 'scores':np.array(srs)}

        # Agregar Dimensión
        if self.add_dim == 3:
            if len(x.shape) == 2:
                # x = x.T
                # x = np.array([x])
                xD = np.empty((3,x.shape[0],x.shape[1]))
                xD[0],xD[1],xD[2] = x,x,x
                # xD =  np.moveaxis(xD,source=0, destination=-1)
                x = xD
            elif len(x.shape) == 3:
                # f = 1
                x = np.moveaxis(x,source=-1, destination=0)
        elif self.add_dim == 2:
            if len(x.shape) == 2:
                x = x.T
                x = np.array([x])
            elif len(x.shape) == 3:
                x = np.moveaxis(x,source=-1, destination=0)
                x = x[0].T
                x = np.array([x])
            # print(x.shape)
            # x = np.moveaxis(x, source=1, destination=-1)
            # x = np.expand_dims(x, axis=0)

        # print('Before: ', target)
        # Encasillar
        if self.tgt_int64:
            x = torch.from_numpy(x).type(torch.float32)
            tgt = {
                key: torch.from_numpy(value).type(torch.int64)
                for key, value in tgt.items()
            }
        else:
            x = torch.from_numpy(x).type(torch.float32)
            tgt = {
                key: torch.from_numpy(value).type(torch.float64)#int64)
                for key, value in tgt.items()
            }
        # print('After: ', target)

        boxes = tgt['boxes']
        labels = tgt['labels']

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = torch.from_numpy(boxes)
        target["labels"] = torch.from_numpy(labels)
        target["image_id"] = image_id
        target["area"] = torch.from_numpy(area)
        target["iscrowd"] = iscrowd
        target["x_name"] = torch.tensor(int(self.inputs[index].name[:-4].replace('_','')))

        return x, target

    @staticmethod
    def read_images(inp, tar):
        return Image.open(inp).convert("RGB"), read_json(tar) #read_pt(tar)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                      #  hidden_layer,
                                                      #  num_classes)

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class MutilabelClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, anno_path, transforms):
        self.transforms = transforms
        with open(anno_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        self.classes = json_data['labels']

        self.imgs = []
        self.annos = []
        self.data_path = data_path
        print('loading', anno_path)
        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annos.append(sample['image_labels'])
        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            vector = [cls in item for cls in self.classes]
            self.annos[item_id] = np.array(vector, dtype=float)
            # print("Item ID: ", item_id,'Item: ', item, 'Vector:', vector, 'Annos: ', self.annos)

    def __getitem__(self, item):
        anno = self.annos[item]
        img_path = os.path.join(self.data_path, self.imgs[item])
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, anno

    def __len__(self):
        return len(self.imgs)

def adapt_data(data, classes, file_name):
    inputs, targets = data
    samples = []
    for i in range(len(targets)):
        samples.append({"image_name":inputs[i].__str__()[-16:],"image_labels":read_json(targets[i])['labels']})
    Js = {"samples":samples, "labels": list(classes.keys())}
    save_json(Js, pathlib.Path(file_name))

def checkpoint_save(model, save_path, epoch, run):
    f = os.path.join(save_path, 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    run["Model States/checkpoint-{:06d}.pth".format(epoch)].upload(f)
    print('saved & uploaded checkpoint:', f)

def show_sample(img, binary_img_labels):
  # Convert the binary labels back to the text representation.
  img_labels = np.array(dataset_val.classes)[np.argwhere(binary_img_labels > 0)[:, 0]]
  plt.imshow(img)
  plt.title("{}".format(', '.join(img_labels)))
  plt.axis('off')
  plt.show()
