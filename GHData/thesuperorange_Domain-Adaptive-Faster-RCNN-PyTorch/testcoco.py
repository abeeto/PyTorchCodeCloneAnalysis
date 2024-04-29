import torch
from maskrcnn_benchmark.data.datasets.coco import COCODataset


ann_path = 'datasets/MI3/annotations/instances_test.json'
#ann_path = 'datasets/MI3_simple.json'
image_root = 'datasets/MI3/JPEGImages'

#ann_path='datasets/cityscapes/annotations/instancesonly_filtered_gtFine_test.json'

#ann_path='datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
#ann_path = 'datasets/foggy_simple.json'
#image_root = 'datasets/cityscapes/images'


#ann_path = 'datasets/coco/annotations/instances_minival2014.json'
#image_root='datasets/coco/val2014'
dataset = COCODataset(ann_path,image_root ,True)
print(len(dataset))
sampler = torch.utils.data.sampler.RandomSampler(dataset)



