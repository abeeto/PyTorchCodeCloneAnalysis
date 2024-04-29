import json
import numpy as np
import glob, pickle

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import skimage.io



annotations = json.load( open('data/v2_mscoco_val2014_annotations.json', 'r') )
questions = json.load( open('data/v2_OpenEnded_mscoco_val2014_questions.json', 'r') )


image2question = {}
for mapping in questions['questions']:
    image2question[ mapping['image_id'] ] = mapping['question']

    
image2answer = {}
for mapping in annotations['annotations']:
    image2answer[ mapping['image_id'] ] = mapping['answers'][0]['answer']


images = glob.glob('data/val2014/*')
print('glob images done')



image2path = {}
for path in images:
    index = int(path.split('.')[0][-6:])
    image2path[index] = path
    


val = []
im2path = list(image2path.keys())

for i in range(len(list(image2path.keys()))):
       
    image_id = im2path[i]
    
    path = image2path[image_id]
    question = image2question[image_id]
    answer = image2answer[image_id]
    if len(answer.split(' ')) == 1 :
        val.append([path, question, answer])



img_transform = transforms.Compose([
                                     transforms.Resize(( 224, 224 )),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                     ])


resnet = torch.load('checkpoint/resnet.pth')
print('Loaded ResNet Model')


def get_feature(path, flip=False):

    img = skimage.io.imread(path)
    if flip: img = np.fliplr(img)
    if(len(img.shape) == 2):
        return False, False
    img = F.to_pil_image(img)
    img = img_transform(img)
    img = img.unsqueeze(0)
    output = resnet(img)
    output = output.view(-1)

    return output.detach().numpy(), True




start = 0

print('LEN = ', len(val))

image_feature = []
for i in range(start, len(val)):
    print('%d / %d' % (i, len(val)))
    path, question, answer = val[i]
    feature, status = get_feature(path)
    #f2, status = get_feature(path, flip=True)
    if status:
        image_feature.append( [ feature, path, question, answer ] )
    else:
        print('Gray-scale image : discarded') 

with open('val_features_'+str(start)+'_'+str(len(val))+'.pkl', 'wb') as f:
    pickle.dump(image_feature, f)


print('\n----------------------------------------------------------------\n')


