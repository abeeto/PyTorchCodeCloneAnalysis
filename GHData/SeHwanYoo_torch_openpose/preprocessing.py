import cv2
import os
import json
import numpy as np
import pandas as pd
from glob import glob
import random

cur_dir = os.path.dirname(os.path.abspath('openpose'))
clients = ['jes', 'khb', 'ljh', 'njw']
skeleton = [
    [1,2],[2,4],[4,3],[3,1], # Head
    [4,5],[4,11],[4,17],[17,18],[17,24], # Body
    [5,6],[6,8],[8,10],[10,9],[9,7],[7,5], # Right arm
    [11,12],[12,14],[14,16],[16,15],[15,13],[13,11], # Left arm
    [18,20],[20,22],[22,23],[23,21],[21,19],[19,18], # Right leg
    [24,25],[25,27],[27,29],[29,28],[28,26],[26,24] # Left leg
]
coco = {

    "info": {
        "description": "",
        "url": "http://bilab.ai/",
        "version": "1.0",
        "year": 2022,
        "contributor": "",
        "data_created": "2022/07/09"
    },

    "licenses": [
        # For example
        # {
        # "url": "http://ycreativecommons.org/licenses/by-nc-sa/2.0/",
        # "id": 1,
        # "name": "Attribution-NonCommercial-ShareAlike License"
        # }
    ],

    "images": [
        # ***Format***
        # {
        #     "license": None,data_dir
        #     "height": 480,
        #     "width": 640,
        #     "date_captured": "2020-08-31T12:43:47.223Z"
        #     "flickr_url": None,
        #     "classId": 1496904
        #     "id": 506412614
    ],

    "categories": [
        {
            "supercategory": "bab",
            "id": 1,
            "name": "GM_Left",
            "keypoints": [
                "Head", "Eye_R", "Eye_L", "Neck", "Sholuder_R", "Elbow_high_R",
                "Elbow_low_R", "Wrist_high_R", "Wrist_low_R", "Hand_R", "Sholuder_L", "Elbow_high_L",
                "Elbow_low_L", "Wrist_high_L", "Wrist_low_L", "Hand_L", "Pelvis", "Hip_R",
                "Knee_right_R", "Knee_left_R", "Ankle_right_R", "Ankle_left_R", "Foot_R", "Hip_L",
                "Knee_right_L", "Knee_left_L", "Ankle_right_L", "Ankle_left_L", "Foot_L"
            ],
            "skeleton": skeleton
        },
        {
            "supercategory": "baby",
            "id": 2,
            "name": "GM_Right",
            "keypoints": [
                "Head", "Eye_R", "Eye_L", "Neck", "Sholuder_R", "Elbow_high_R",
                "Elbow_low_R", "Wrist_high_R", "Wrist_low_R", "Hand_R", "Sholuder_L", "Elbow_high_L",
                "Elbow_low_L", "Wrist_high_L", "Wrist_low_L", "Hand_L", "Pelvis", "Hip_R",
                "Knee_right_R", "Knee_left_R", "Ankle_right_R", "Ankle_left_R", "Foot_R", "Hip_L",
                "Knee_right_L", "Knee_left_L", "Ankle_right_L", "Ankle_left_L", "Foot_L"
            ],
            "skeleton": skeleton
        }
    ],

    "annotations": [
        # ***Format***
        # {
        #     "segmentation": None,
        #     "num_keypoints": 29,
        #     "area": None,
        #     "iscrowd": None,
        #     "keypoints": [],
        #     "image_id": 506412614,
        #     "bbox": [],
        #     "category_id": int,
        #     "id": None
        # }
    ]

}


def find_bbox(img, old_bbox):
# Finds bounding box from keypoints using canny edge detection

    h, w = img.shape[:2]
    gap = 14

    old_xmin, old_xmax, old_ymin, old_ymax = old_bbox
    xmin = max(0, old_xmin-gap)
    xmax = min(w, old_xmax-gap)
    ymin = max(0, old_ymin-gap)
    ymax = min(h, old_ymax+gap)

    # Erase out the background far from the object's bounding box
    img_cropped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Otsu's threshold for canny edge detection
    th, ret = cv2.threshold(img_cropped, 0, 255, cv2.THRESH_OTSU)

    canny_output = cv2.Canny(img_cropped, th, th*2)
    canny_output[:int(ymin), :] = 0
    canny_output[int(ymax):, :] = 0
    canny_output[:, :int(xmin)] = 0
    canny_output[:, int(xmax):] = 0

    # Find the bounding box
    x,y,w,h = cv2.boundingRect(canny_output)
    x = min(x, old_xmin)
    y = min(y, old_ymin)
    w = max(w, old_xmax-x)
    h = max(h, old_ymax-y)

    # Visualize edges with bounding box
    # cv2.rectangle(canny_output, (int(x),int(y)), (int(x+w),int(y+h)), 255, 5)
    # cv2.imwrite(os.path.join(cur_dir, 'contour.jpg'), canny_output)

    return [x,y,w,h]

def datalist_split(data, rate=0.1):
    length = int(len(data) * rate)
    return data[:length], data[length:]

def preprocessing(data_dir, datalist_name='train'):
    idx = 0
    for ann in data_dir:
        img = cv2.imread(ann.replace('ann', 'img').rstrip('.json'))
        with open(ann) as f:
            data = json.load(f)
            if not data["objects"]:
                print("No object")
                continue

            obj = data["objects"][0]
            file_name = str(idx).zfill(6) + '.png'
            id = obj["id"] # image id 
            classId = obj["classId"]

            classTitle = obj["classTitle"]
            if classTitle == "GM_Left":
                category_id = 1
            elif classTitle == "GM_Right":
                category_id = 2
            else:
                continue

            keypointsX = []
            keypointsY = []
            keypoints = []
            num_nodes = len(obj["nodes"])
            assert num_nodes == num_keypoints, f"{ann_file} has {num_nodes} nodes"

            nodes = obj["nodes"]
            first_key = list(nodes.keys())[0]
            node_table = []
            for df in GM_node_excel:
                if first_key in df.nodes.values:
                    node_table = df.nodes.values
                    break


            if len(node_table) == 0:
                print("Node key is not in the node table list")
                continue
            
            try:
                for key in node_table:
                    keypointsX.append(nodes[key]["loc"][0])
                    keypointsY.append(nodes[key]["loc"][1])

                    # 2022.08.17 modifed shyoo 
                    # keypoints.append(nodes[key]["loc"])
                    keypoints.extend(nodes[key]["loc"])
                    """
                    Annotations for keypoints is specified in (x, y, v)
                    v indicates visibility
                    v=0: not labeled (x=y=0)
                    v=1: labeled but not visible
                    v=2: labeled and visible
                    """
                    # 2022.08.17 modifed shyoo 
                    keypoints.append(2)
                    # keypoints.extend(2)
                    
            except KeyError:
                continue

            xmin = min(keypointsX)
            xmax = max(keypointsX)
            ymin = min(keypointsY)
            ymax = max(keypointsY)

            bbox = find_bbox(img, [xmin, xmax, ymin, ymax])

            coco["images"].append(
                {
                    "license": None,
                    "file_name": file_name,
                    "coco_url": None,
                    "height": data["size"]["height"],
                    "width": data["size"]["width"],
                    "date_captured": obj["createdAt"],
                    "flickr_url": None,
                    "id": id
                }
            )

            coco["annotations"].append(
                {
                    "segmentation": None,
                    "num_keypoints": num_keypoints,
                    "area": None,
                    "iscrowd": None,
                    "keypoints": keypoints,
                    "image_id": id,
                    "bbox": bbox,
                    "category_id": 1,
                    "id": None
                }
            )                    
                
            # Save images and annotated images
            bbox = list(map(int, bbox))
            keypointsX = list(map(int, keypointsX))
            keypointsY = list(map(int, keypointsY))

            if not os.path.exists(os.path.join('custom_dataset/images', datalist_name)):
                os.mkdir(os.path.join('custom_dataset/images', datalist_name))

            if not os.path.exists(os.path.join('custom_dataset/annotated images', datalist_name)):
                os.mkdir(os.path.join('custom_dataset/annotated images', datalist_name))


            cv2.imwrite(os.path.join('custom_dataset/images', datalist_name, file_name), img)
            cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 3)
            for start, end in skeleton:
                cv2.line(img, (keypointsX[start-1], keypointsY[start-1]), (keypointsX[end-1], keypointsY[end-1]), (255,0,0), 3)
            cv2.imwrite(os.path.join('custom_dataset/annotated images', file_name), img)

        idx += 1

    # return coco
    with open(os.path.join('custom_dataset/annotations', f'{datalist_name}_baby_keypoints.json'), 'w') as f:
        json.dump(coco, f)


# Read GM_node.xlsx
GM_node_excel = pd.read_excel(os.path.join(cur_dir, 'Dataset/GM_node.xlsx'), sheet_name=None).values()


last_idx = 23528
num_keypoints = 29


train_list = [] 
valid_list = [] 
test_list = [] 

for c in clients:
    client_dir = os.path.join(cur_dir, 'Dataset', c)
    client_list = glob(os.path.join(client_dir, '*/*/ann/*.json'))

    train, test = datalist_split(client_list, 0.6)    
    train_list.extend(train)

    test, valid = datalist_split(test, 0.5)
    test_list.extend(test)
    valid_list.extend(valid)

# preprocessing
preprocessing(train_list, 'train')
preprocessing(valid_list, 'valid')
preprocessing(test_list, 'test')