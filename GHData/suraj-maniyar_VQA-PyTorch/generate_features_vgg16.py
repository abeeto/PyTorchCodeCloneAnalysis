import json
import numpy as np
import glob, pickle, os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.models as models

import skimage.io

os.environ['TORCH_HOME'] = '/share/corticalproc/spmaniya/VQA/checkpoint'



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
    


train = []
im2path = list(image2path.keys())

for i in range(len(list(image2path.keys()))):
       
    image_id = im2path[i]
    
    path = image2path[image_id]
    question = image2question[image_id]
    answer = image2answer[image_id]
    if len(answer.split(' ')) == 1 :
        train.append([path, question, answer])



img_transform = transforms.Compose([
                                     transforms.Resize(( 224, 224 )),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                     ])






class VGG16(nn.Module):

     def __init__(self):

         super().__init__()
         
         self.vgg16 = models.vgg16(pretrained=True)   
         self.layer1 = nn.Sequential(*list(self.vgg16.features.children())) 
         self.layer3 = nn.Sequential(*list(self.vgg16.classifier.children())[0:4])
   
     def forward(self, x):
         out = self.layer1(x)
         out = out.view(out.size(0),-1)
         out = self.layer3(out)
        
         return out 
     

vgg16 = VGG16()

if torch.cuda.is_available():
    vgg16 = vgg16.cuda()

#vgg16 = torch.load('checkpoint/vgg16.pth')
print('Loaded VGG16 Model')








def get_feature(path, flip=False):

    img = skimage.io.imread(path)
    if flip: img = np.fliplr(img)
    if(len(img.shape) == 2):
        return False, False
    img = F.to_pil_image(img)
    img = img_transform(img)
    img = img.unsqueeze(0)
    
    img = img.cuda()

    output = vgg16(img)
    output = output.view(-1)

    return output.cpu().detach().numpy(), True


print('LEN = ', len(train))




def save(start, end):
    image_feature = []
    for i in range(start, end):
        print('%d / %d' % (i, len(train)))
        path, question, answer = train[i]
        f1, status = get_feature(path)
        #f2, status = get_feature(path, flip=True)
        if status:
            image_feature.append( [ f1, path, question, answer ] )
        else:
            print('Gray-scale image : discarded') 

    with open('dumps/val_features_vgg16_'+str(start)+'_'+str(end)+'.pkl', 'wb') as f:
        pickle.dump(image_feature, f)


    print('\n----------------------------------------------------------------\n')


save(30000, len(train))
