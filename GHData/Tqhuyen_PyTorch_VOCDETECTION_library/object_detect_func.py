import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
# torchvision.transforms.functional.convert_image_dtype

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



def show_image_and_label(image: torch.Tensor, label: dict)->torch.Tensor:
  '''
  input:
  image: shape=[C,H,W], type: torch.Tensor
  label: type: dict
  structure:
  
  {'annotation': {'filename': '000044.jpg',
                'folder': 'VOC2007',
                'object': [{'bndbox': {'xmax': '370',
                                       'xmin': '1',
                                       'ymax': '330',
                                       'ymin': '1'},
                            'difficult': '0',
                            'name': 'chair',
                            'pose': 'Unspecified',
                            'truncated': '1'},
                           {'bndbox': {'xmax': '312',
                                       'xmin': '99',
                                       'ymax': '213',
                                       'ymin': '101'},
                            'difficult': '0',
                            'name': 'cat',
                            'pose': 'Right',
                            'truncated': '0'}],
                'owner': {'flickrid': 'Urban Echo', 'name': '?'},
                'segmented': '0',
                'size': {'depth': '3', 'height': '333', 'width': '500'},
                'source': {'annotation': 'PASCAL VOC2007',
                           'database': 'The VOC2007 Database',
                           'flickrid': '340274411',
                           'image': 'flickr'}}}

  this func write specially for VOCDetection2007
  Output:

  result : [C,H,W], dtype: Tensor
  '''
  
  number_of_object = len(label['annotation']['object'])
  boxes = []
  names = []
  for box_index in range(number_of_object):
      xmax = label['annotation']['object'][box_index]['bndbox']['xmax']
      xmin = label['annotation']['object'][box_index]['bndbox']['xmin']
      ymax = label['annotation']['object'][box_index]['bndbox']['ymax']
      ymin = label['annotation']['object'][box_index]['bndbox']['ymin']
      boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
      name = label['annotation']['object'][box_index]['name']
      names.append(name)
  colors = ["blue", "yellow"]
  boxes = torch.tensor(boxes, dtype=torch.float)
  image = F.convert_image_dtype(image, dtype=torch.uint8)
  result = draw_bounding_boxes(image=image, boxes=boxes, labels=names, colors=colors, width=2)
  return result
