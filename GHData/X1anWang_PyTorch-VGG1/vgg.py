import torch
import torchvision
import cv2
from PIL import Image
import os
import numpy as np
import time

if __name__ == '__main__':
    list_device = ['cuda', 'cpu']
    path = '/home/xian/dataset/hymenoptera_data/'
    candidate = ['bees', 'ants']
    label = []
    inputs = []
    trans = torchvision.transforms.Compose([
                    torchvision.tranforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
    for species in candidate:
        pathh = path + species + '/'
        input_size = 224
        filenames = []
        if (species == 'bees'):
            filenames = os.listdir(r'/home/xian/dataset/hymenoptera_data/bees/')
            label = label + [309] * len(filenames)
        else:
            filenames = os.listdir(r'/home/xian/dataset/hymenoptera_data/ants/')
            label = label + [310] * len(filenames)
        for names in filenames:
            inputs = inputs + [(pathh+names)]
    
    for devi in list_device:
        print(devi)
        device = torch.device(devi)
        model = torchvision.models.vgg16_bn(pretrained=True)
        model = model.to(device)
        model.eval()
        acc = 0
        total = 0

        start = time.time()
        for i in range(len(inputs)):
            try:
                img = cv2.imread(inputs[i])
                img = cv2.resize(img, (224, 224))
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                tensor = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()/255.0
                tensor = tensor.reshape((1, 3, 224, 224))
                tensor = trans(tensor)
                tensor = tensor.to(device)
                output = model(tensor)
                _, pred = torch.max(output.data, 1)
                print("OutPut Label: ")
                print(str(pred.item()))
                if pred.item() == label[i]:
                    acc = acc + 1
            except:
                print("pass!")
                pass
        stop = time.time()
        print("Time elapsed:...")
        print(str(stop-start))
        print("Total: "+str(total))
        print("Accuracy: "+str(acc))