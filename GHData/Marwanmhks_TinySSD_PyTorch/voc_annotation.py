import os
import random
import xml.etree.ElementTree as ET

from utils.tools import get_classes

# ------------------------------------------------- --------------------------------------------------
# annotation_mode is used to specify what is calculated when the file is run
# annotation_mode is 0 to represent the entire label processing process, including obtaining txt in
# VOCdevkit/VOC2007/ImageSets and 2007_train.txt, 2007_val.txt for training annotation_mode is 1 to get the txt in
# VOCdevkit/VOC2007/ImageSets annotation_mode is 2 to obtain 2077_train.txt, 2077_val.txt for training
# ---------------------------------------------------------------------------------------------------
annotation_mode = 0

# --------------------------------------------------------------------
# trainval_percent is used to specify the
# ratio of (training set + validation set) to test set, by default (training set + validation set): test set = 9:1
# train_percent is used to specify the ratio of training set to validation set in (training set + validation set),
# by default training set:validation set = 9:1
# --------------------------------------------------------------------
trainval_percent = 0.9
train_percent = 0.9

Dir_path = 'C:\\Users\\Marwan\\PycharmProjects\\TinySSD_Banana\\TinySSD_Banana'
classes_path = os.path.join(Dir_path, 'model_data\\voc_classes.txt')
classes, _ = get_classes(classes_path)

# -------------------------------------------------
# Point to the folder where the VOC dataset is located
# Default points to the VOC dataset in the root directory
# -------------------------------------------------
year = 2077
VOCdevkit_path = os.path.join(Dir_path, 'VOCdevkit')
VOCdevkit_sets = [(f'{year}', 'train'), (f'{year}', 'val'), (f'{year}', 'test')]


def convert_annotation(_year, _image_id, _list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s\\Annotations\\%s.xml' % (_year, _image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        _list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets\\Main ...")
        xmlfilepath = os.path.join(VOCdevkit_path, f'VOC{VOCdevkit_sets[0][0]}\\Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, f'VOC{VOCdevkit_sets[0][0]}\\ImageSets\\Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        itemList = range(num)

        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(itemList, tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in itemList:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print(f"Generate .txt for train...")
        for year, image_set in VOCdevkit_sets:
            # image_ids saved the name of imgs(or annotations)(don't include suffix)
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s\\ImageSets\\Main\\%s.txt' % (year, image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s\\VOC%s\\JPEGImages\\%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("done\n")
