import os
import argparse
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
import random
import shutil


def crop_and_resize(img, bbox_list):
    crop_img_list = []

    for bbox in bbox_list:
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crop_img = cv2.resize(crop_img, (112, 112))
        crop_img_list.append(crop_img)

    return crop_img_list


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    person_list = []

    for object in root.findall('object'):
        bndbox = object.find('bndbox')
        name = object.find('name').text
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox = [xmin, ymin, xmax, ymax]

        person_dict = {
            'name': name,
            'bbox': bbox
        }
        person_list.append(person_dict)

    return person_list


def get_anno_list(args):
    raw_path = args.raw_path
    folder_list = os.listdir(raw_path)

    img_path_list = []
    label_path_list = []

    for folder in folder_list:
        folder_full_path = os.path.join(raw_path, folder)
        file_list = os.listdir(folder_full_path)

        img_path_list += [os.path.join(folder_full_path, x) for x in file_list if "jpg" in x]
        label_path_list += [os.path.join(folder_full_path, x) for x in file_list if "xml" in x]

    return img_path_list, label_path_list


def preprocess_face_img(args, img_path_list, label_path_list):
    total_face = 0

    for label_path in tqdm(label_path_list):
        img_path = label_path.replace('xml', 'jpg')
        person_list = read_xml(label_path)

        name_list = [x['name'] for x in person_list]
        bbox_list = [x['bbox'] for x in person_list]
        img = cv2.imread(img_path)
        crop_img_list = crop_and_resize(img, bbox_list)

        for idx, name in enumerate(name_list):
            crop_img = crop_img_list[idx]
            crop_img_name = '{}_{}.jpg'.format(img_path.split('/')[-1].split('.')[0], idx)

            if not os.path.exists(os.path.join(args.out_path, name)):
                os.makedirs(os.path.join(args.out_path, name))

            cv2.imwrite(os.path.join(args.out_path, name, crop_img_name), crop_img)
            total_face += 1

    print('total face img : {}'.format(total_face))


def preprocess(args):
    img_path_list, label_path_list = get_anno_list(args)
    preprocess_face_img(args, img_path_list, label_path_list)


def split_train_test(args):
    test_ratio = args.test_ratio

    out_path = args.out_path
    folder_list = os.listdir(out_path)

    print("split dataset")
    for folder in tqdm(folder_list):
        if folder == "None":
            continue

        folder_full_path = os.path.join(out_path, folder)
        file_list = os.listdir(folder_full_path)
        random.shuffle(file_list)

        if len(file_list) * test_ratio < 1.0:
            if len(file_list) == 1:
                test_file_list = file_list
                train_file_list = file_list

            else:
                test_file_list = file_list[:1]
                train_file_list = file_list[1:]

        else:
            test_file_list = file_list[:int(test_ratio * len(file_list))]
            train_file_list = file_list[int(test_ratio * len(file_list)):]

        # test split
        if not os.path.exists(os.path.join(args.test_path, folder)):
            os.makedirs(os.path.join(args.test_path, folder))

        for test_file in test_file_list:
            shutil.copyfile(os.path.join(args.out_path, folder, test_file), os.path.join(args.test_path, folder, test_file))

            if folder != "background":
                no_bg_path = "{}_no_bg".format(args.test_path)

                if not os.path.exists(os.path.join(no_bg_path, folder)):
                    os.makedirs(os.path.join(no_bg_path, folder))

                shutil.copyfile(os.path.join(args.out_path, folder, test_file),
                                os.path.join(no_bg_path, folder, test_file))


        # train_split
        if not os.path.exists(os.path.join(args.train_path, folder)):
            os.makedirs(os.path.join(args.train_path, folder))

        for train_file in train_file_list:
            shutil.copyfile(os.path.join(args.out_path, folder, train_file), os.path.join(args.train_path, folder, train_file))

            if folder != "background":
                no_bg_path = "{}_no_bg".format(args.train_path)

                if not os.path.exists(os.path.join(no_bg_path, folder)):
                    os.makedirs(os.path.join(no_bg_path, folder))

                shutil.copyfile(os.path.join(args.out_path, folder, train_file),
                                os.path.join(no_bg_path, folder, train_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Crop and resize face")
    parser.add_argument("--raw_path", type=str, default="../love_war_dataset/raw_data")
    parser.add_argument("--out_path", type=str, default="../love_war_dataset/face_img_folder")
    parser.add_argument("--train_path", type=str, default="../love_war_dataset/train_face_img_folder")
    parser.add_argument("--test_path", type=str, default="../love_war_dataset/test_face_img_folder")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()
    preprocess(args)
    split_train_test(args)
