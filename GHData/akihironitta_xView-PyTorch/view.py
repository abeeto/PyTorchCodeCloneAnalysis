# Copyright (c) 2020 Akihiro Nitta
# All rights reserved.
# 
import argparse
import os
import cv2
import numpy as np
from datasets import XviewDataset, LABEL_TO_STRING
from bbox_visualizer import draw_rectangle, add_label_to_rectangle

try:
    import colored_traceback.always  # noqa
except ImportError:
    pass


DEFAULT_IMAGE_DIR = "/home/nitta/data/xview/train_images/"
DEFAULT_ANNOTATION_FILE = "/home/nitta/data/xview/xView_train.geojson"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", nargs="*", type=int)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--root", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotation", type=str, default=DEFAULT_ANNOTATION_FILE)
    return parser.parse_args()


def draw_bboxes(img, target):
    for t in target:
        x, y, w, h = t["bbox"]
        bbox = [x, y, x+w, y+h]     # [x_min, y_min, x_max, y_max]
        img = draw_rectangle(img, bbox)
        if t["category_id"] in LABEL_TO_STRING.keys():
            label = LABEL_TO_STRING[t["category_id"]]
        else:
            label = str(t["category_id"])
        img = add_label_to_rectangle(img, label, bbox)
    return img


def show(img, key="q"):
    while True:
        cv2.imshow("output", img)
        k = cv2.waitKey(1)
        if k == ord(key):
            break
    cv2.destroyAllWindows()

    
def save(img, fname):
    cv2.imwrite(fname, img)

    
def main():
    args = parse_args()
    ds = XviewDataset(root=args.root, annFile=args.annotation)
    
    index_str, image_str, n_objects_str = "index", "image", "n_objects"
    print(f"| {index_str:^{len(index_str)}} | {image_str:^{len(image_str)}} | {n_objects_str:^{len(n_objects_str)}} |")
    print(f"|{'-'*(len(index_str)+2)}|{'-'*(len(image_str)+2)}|{'-'*(len(n_objects_str)+2)}|")
    
    for idx in args.index:
        img, target = ds[idx]
        img_id = target[0]["image_id"]
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_bboxes(img, target)
        
        print(f"| {idx:^{len(index_str)}} | {img_id:{len(image_str)}} | {len(target):{len(n_objects_str)}} |")
        
        if args.output:
            fname = os.path.join(args.output, str(img_id)+"-bbox.tif")
            save(img, fname)
        else:
            show(img)
        

if __name__ == "__main__":
    main()
