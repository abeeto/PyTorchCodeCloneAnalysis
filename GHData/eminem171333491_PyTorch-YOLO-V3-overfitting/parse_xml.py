from xml.dom.minidom import parse
import os
import numpy as np
import re


classname = {"人类":0,"大熊猫":1,"小熊猫":2,"浣熊":3}
# print(classname["人"])

txt_path = r"./data/"
txt = open(txt_path + "the_label.txt", "w",encoding="utf-8")

xml_path = r"./data/img/outputs"
for xml in os.listdir(xml_path):
    dom = parse(os.path.join(xml_path,xml))
    root = dom.documentElement
    img_name = root.getElementsByTagName("path")[0].childNodes[0].data
    # print(root.getElementsByTagName("path")[0].childNodes[0].data)
    img_size= root.getElementsByTagName("size")[0]
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    # objects = root.getElementsByTagName("object")
    # print(img_name)
    # print(img_w,img_h,img_c)
    # for boxes in objects:
    #     item = boxes.getElementsByTagName("item")
    #     for box in item:
    #         cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
    #         x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
    #         y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
    #         x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
    #         y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
    #         print(cls_name,x1,y1,x2,y2)
    item = root.getElementsByTagName("item")
    txt.write(str(img_name) + " ")
    for box in item:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
        y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
        cx = (x1 + x2)*0.5
        cy = (y1 + y2)*0.5
        w = x2 - x1
        h = y2 - y1
        print(cls_name,x1,y1,w,h)
        txt.write(" {} {} {} {} {}".format(classname[cls_name], cx, cy, w, h))
    txt.write("\n")




