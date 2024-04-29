from commons import *
from input_processing import transform
# XML PARSING
import xml.etree.ElementTree as ET
import torch.utils.data as data

from PIL import Image
from prettytable import PrettyTable

object_count_by_type = {}



def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()

        if label not in label_map:
            print("{} Not in LabelMap".format(label))
            continue
        else:
            if label in object_count_by_type.keys():
                object_count_by_type[label] += 1
            else:
                object_count_by_type[label] = 1

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)-1
        ymin = int(bbox.find('ymin').text)-1
        xmax = int(bbox.find('xmax').text)-1
        ymax = int(bbox.find('ymax').text)-1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes':boxes, 'labels':labels, 'difficulties':difficulties}


def parse_data(voc07_data, voc12_data, output_folder):
    voc07_data = os.path.abspath(voc07_data)
    voc12_data = os.path.abspath(voc12_data)

    train_images = list()
    train_objects = list()
    n_objects = 0


    for path in [voc07_data, voc12_data]:
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            objects = parse_annotation(os.path.join(path, 'Annotations',id+'.xml'))

            if len(objects['boxes']) == 0:
                continue

            n_objects += len(objects['boxes'])
            train_objects.append(objects)
            train_images.append(os.path.join(path,'JPEGImages',id+'.jpg'))

        assert len(train_images) == len(train_objects)


    with open(os.path.join(output_folder,"TRAIN_images.json"), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder,"TRAIN_objects.json"), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder,"label_map.json"), 'w') as j:
        json.dump(label_map, j)

    print("\nThere are {} training images containing a total of {} objects.\n".format(len(train_images), n_objects))

    test_images = list()
    test_objects = list()
    n_objects = 0

    path = voc07_data # we only have test sets available for voc07 dataset

    with open(os.path.join(path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        objects = parse_annotation(os.path.join(path, 'Annotations',id+'.xml'))

        if len(objects['boxes']) == 0:
            continue

        n_objects += len(objects['boxes'])
        test_objects.append(objects)
        test_images.append(os.path.join(path,'JPEGImages',id+'.jpg'))

    assert len(test_images) == len(test_objects)


    with open(os.path.join(output_folder,"TEST_images.json"), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder,"TEST_objects.json"), 'w') as j:
        json.dump(test_objects, j)


    print("\nThere are {} test images containing a total of {} objects.\n".format(len(test_images), n_objects))

    t = PrettyTable(['Object', 'Count'])
    for key in sorted(list(object_count_by_type.keys())):
        t.add_row([key, object_count_by_type[key]])
    print(t)


class PascalGeneratorDataset(data.Dataset):
    def __init__(self, data_folder, split):
        self.split = split.upper()

        assert self.split in {"TEST", "TRAIN"}

        self.data_folder = data_folder

        with open(os.path.join(data_folder,self.split+"_images.json"), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder,self.split+"_objects.json"), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        objects = self.objects[i] # this is a map/dictionary of boxes, labels and difficulties

        boxes = torch.FloatTensor(objects['boxes'])
        labels = torch.LongTensor(objects['labels'])
        difficults = torch.BoolTensor(objects['difficulties'])

        return self.images[i], [boxes], [labels], [difficults]



class PascalDataset(data.Dataset):
    def __init__(self, data_folder, split, keep_difficult=True, resize_dims=(300,300)):
        self.split = split.upper()

        assert self.split in {"TRAIN", "TEST"}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        self.resize_dims = resize_dims

        with open(os.path.join(data_folder,self.split+"_images.json"), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder,self.split+"_objects.json"), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        objects = self.objects[i] # this is a map/dictionary of boxes, labels and difficulties

        boxes = torch.FloatTensor(objects['boxes'])
        labels = torch.LongTensor(objects['labels'])
        difficults = torch.BoolTensor(objects['difficulties'])

        if not self.keep_difficult:
            boxes = boxes[~difficults]
            labels = label_map[~difficults]
            difficult = difficult[~difficults]

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficults, split=self.split, resize_dims=self.resize_dims) # converts all but difficults to tensors

        return image, boxes, labels, difficulties
    def collate_fn(self, batch):
        all_images = list()
        all_boxes = list()
        all_labels = list()
        all_difficults = list()

        for b in batch:
            all_images.append(b[0])
            all_boxes.append(b[1])
            all_labels.append(b[2])
            all_difficults.append(b[3])

        all_images = torch.stack(all_images, dim=0)

        return all_images, all_boxes, all_labels, all_difficults
