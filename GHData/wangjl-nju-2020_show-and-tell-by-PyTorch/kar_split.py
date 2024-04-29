"""
Karpathy Split for MS-COCO Dataset
合并原始train val集以获得更多的训练数据，保留val中的5000个作为新val，保留test中的5000个作为新test
生成三个集合的json文件,保存在data_root/karpathy_split下
"""
import os
import json
from random import shuffle, seed
from config import DATA_ROOT

seed(0)  # Make it reproducible

num_val = 5000
num_test = 5000

train_path = DATA_ROOT + 'annotations/captions_train2014.json'
val_path = DATA_ROOT + 'annotations/captions_val2014.json'
val = json.load(open(train_path, 'r'))
train = json.load(open(val_path, 'r'))

# Merge together
imgs = val['images'] + train['images']
annotations = val['annotations'] + train['annotations']

shuffle(imgs)

# Split into val, test, train
dataset = {'val': imgs[:num_val],
           'test': imgs[num_val: num_val + num_test],
           'train': imgs[num_val + num_test:]}

# Group by image ids
img2ann = {}  # 以image_id查找img2ann信息的词典，包括全部数据
for annotation in annotations:
    img_id = annotation['image_id']
    if img_id not in img2ann:
        img2ann[img_id] = []

    img2ann[img_id].append(annotation)

json_data = {}
info = train['info']
licenses = train['licenses']

split = ['val', 'test', 'train']

for subset in split:
    # images存储所有图片的img信息，annotations存储annotations的信息
    json_data[subset] = {'type': 'caption',
                         'info': info,
                         'licenses': licenses,
                         'images': [],
                         'annotations': []}

    # 生成结果是每个子集划分img_id， 根据img_id划分所有的annotation
    for img in dataset[subset]:
        img_id = img['id']
        anns = img2ann[img_id]

        json_data[subset]['images'].append(img)
        json_data[subset]['annotations'].extend(anns)

    # 写为三个json文件
    if not os.path.exists(DATA_ROOT + 'karpathy_split/'):
        os.makedirs(DATA_ROOT + 'karpathy_split/')

    with open(DATA_ROOT + 'karpathy_split/' + subset + '.json', 'w') as f:
        json.dump(json_data[subset], f)

print(f"train num: {len(json_data['train']['images'])}")
print(f"val num: {len(json_data['val']['images'])}")
print(f"test num: {len(json_data['test']['images'])}")
print('*** Complete ***')
