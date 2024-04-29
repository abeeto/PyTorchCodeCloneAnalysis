import os
import json
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    size = tree.find('size')
    w, h = size.find('width').text, size.find('height').text
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects, h, w

def coco_instances(voc_root):
  data = {
    "info": {"description": "VOC2007"},
    "licenses": [{"company": "corerain", "id": 1}],
    "type": "instances",
  }
  categories = []
  for index, cls_name in enumerate(VOC_CLASSES):
    categories.append({
      'id': index + 1,
      'name': cls_name,
    })
  data['categories'] = categories
  images = []
  annotations = []
  with open('./data/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      file_name = line.strip()
      try:
        objects, h, w = parse_rec(os.path.join(voc_root, file_name + '.xml'))
      except FileNotFoundError as e:
        continue
      
      images.append({
        "license": 1,
        "id": i,
        "file_name": file_name + '.jpg',
        "width": w,
        "height": h
      })
      for obj in objects:
        annotations_id = len(annotations)
        xmin, ymin, xmax, ymax = obj['bbox']
        box_w = xmax - xmin
        box_h = ymax - ymin
        area = box_w * box_h
        annotations.append({
          "id": annotations_id,
          "category_id": VOC_CLASSES.index(obj['name']),
          "image_id": i,
          "iscrowd": 0,
          "area": area,
          "bbox": [xmin, ymin, box_w, box_h]
        })

  data['images'] = images
  data['annotations'] = annotations
  with open('instances_voc2007.json', 'w') as f:
    f.write(json.dumps(data))


if __name__ == '__main__':
    coco_instances('./data/VOCdevkit/VOC2007/Annotations')