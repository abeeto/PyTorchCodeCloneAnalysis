import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from model.tiny_ssd import Tiny_SSD
from utils.tools import get_classes
from utils.utils_map import get_map

if __name__ == "__main__":
    '''
        Unlike AP, Recall and Precision are a concept of area. When the threshold value is different, the Recall and Precision values of the network are different.
        The Recall and Precision in the map calculation result represent the corresponding Recall and Precision values when the threshold confidence is 0.5 during prediction.

        The number of txt boxes in ./map_out/detection-results/ obtained here will be more than the direct predict, because the threshold here is low,
        The purpose is to calculate the Recall and Precision values under different threshold conditions, so as to realize the calculation of the map.
        '''
    # ------------------------------------------------- -------------------------------------------------- ---------------#
    # map_mode is used to specify what is calculated when the file is run
    # map_mode is 0 means only get the real box.
    # map_mode is 1 to represent [obtain prediction results, calculate VOC_map]
    # map_mode of 2 means only the prediction result is obtained.
    # map_mode is 3 to only calculate VOC_map.
    # ------------------------------------------------- -------------------------------------------------- ----------------#
    mode = 1
    # ------------------------------------------------- ------#
    # The classes_path here is used to specify the categories that need to measure the VOC_map
    # In general, it can be consistent with the classes_path used for training and prediction
    # ------------------------------------------------- ------#
    Dir_path = 'C:\\Users\\Marwan\\PycharmProjects\\TinySSD_Banana\\TinySSD_Banana'
    classes_path = os.path.join(Dir_path, 'model_data\\voc_classes.txt')
    # ------------------------------------------------- ------#
    # MINOVERLAP is used to specify the mAP0.x you want to get
    # For example, to calculate mAP0.75, you can set MINOVERLAP = 0.75
    # ------------------------------------------------- ------#
    MINOVERLAP = 0.5
    # ------------------------------------------------- ------#
    # Point to the folder where the VOC dataset is located
    # Default points to the VOC dataset in the root directory
    # ------------------------------------------------- ------#
    VOCdevkit_path = 'C:\\Users\Marwan\PycharmProjects\TinySSD_Banana\TinySSD_Banana\VOCdevkit'
    year = '2077'
    # ------------------------------------------------- ------#
    # The folder for the result output, the default is result
    # ------------------------------------------------- ------#
    map_out_path = 'result'

    image_ids = open(os.path.join(VOCdevkit_path, f"VOC{year}\ImageSets\Main\\test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if mode == 0:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth\\" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, f"VOC{year}\Annotations\\" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if mode == 1 or mode == 2:
        print("Load model ...")
        net = Tiny_SSD()

        print("Get predict result ...")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, f"VOC{year}\JPEGImages\\" + image_id + ".jpg")
            image = Image.open(image_path)
            # if map_vis:
            #     image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            net.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if mode == 1 or mode == 3:
        print("Get map ...")
        mAP = get_map(MINOVERLAP, True, path=map_out_path)

        print("Save image results ...")
        if mAP > 0.8:
            from PIL import ImageDraw, ImageFont

            font = ImageFont.truetype('Sundries\simhei.ttf', 20)
            for image_id in tqdm(image_ids):
                image_path = os.path.join(VOCdevkit_path, f"VOC{year}\JPEGImages\\" + image_id + ".jpg")
                image = Image.open(image_path)
                painter = ImageDraw.ImageDraw(image)

                with open(os.path.join(map_out_path, f'ground-truth\\{image_id}.txt')) as f:
                    gt_lines = f.readlines()
                with open(os.path.join(map_out_path, f'detection-results\\{image_id}.txt')) as f:
                    pred_lines = f.readlines()

                for i, line in enumerate(gt_lines): 
                    line = line.split()
                    x1, y1, x2, y2 = list(map(int, line[1:]))
                    painter.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red', width=5)
                    painter.text((x1, y1), line[0], font=font)
                for i, line in enumerate(pred_lines):
                    line = line.split()
                    if float(line[1]) < 0.3:
                        continue 
                    x1, y1, x2, y2 = list(map(int, line[2:]))
                    painter.rectangle(((x1, y1), (x2, y2)), fill=None, outline='blue', width=1)
                    painter.text((x1, y1), f'{line[0]}:[{line[1]}]', font=font)

                image.save(os.path.join(map_out_path, f'images-optional\\{image_id}.jpg'))
        print('Done.')
