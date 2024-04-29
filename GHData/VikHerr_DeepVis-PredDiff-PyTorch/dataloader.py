
import torch 
import torchvision as tv
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt

class DataLoader:

    def __init__(self, path='./data/val/', img_dims=(3,224,224), ):
        '''
        init data loader for MobileNetV2 (for now only MNV2!)
        preprocess: list [<preprocess function>, <display function>] of specific model
        '''

        self.preprocess = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        
        self.display = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor()])
        

        self.data_path = path
        self.img_dims = img_dims


    def laodImage(self, path):
        '''
        load image from file and create format expected by the model
        '''

        input_image = Image.open(path) 
        #input_image.show()
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        
        return input_batch, input_image, path

    def get_data(self,s_idx=1,b_size=4,set_size=None, sort=True):
        '''
        
        ### get data from data set
        paramter: 
        - s_idx    : offset for startpoint in dataset
        - b_size   : number of samples for output
        - set_size : number of considered samples (some might be deleted 
        when they do not fit into a 224x224 image) this should be set for really large data sets
        - sort     : sort os dir list before access (this might be really slow for big data set)
        '''

        img_set = os.listdir(self.data_path)
        if sort:
            img_set.sort()
        # print(img_set)

        # restrict cadidates
        if set_size:
            img_set = img_set[s_idx:(s_idx + set_size + 1)]
        else:
            img_set = img_set[s_idx:]

        image_list = img_set # [1:set_size]
        batch_paths = []
        batch_dspl  = []


        print('Using dataset:', self.data_path)
        # empty batch
        batch = torch.empty((0,self.img_dims[0], self.img_dims[1], self.img_dims[2]))
        batch_end = b_size

        for idx, img_path in enumerate(image_list):
        # for idx in b_size:

            #img_path = 'ILSVRC2012_val_00013083.JPEG'
            img = Image.open(self.data_path + img_path)
            y,x = img.size
            ch  = len(img.getbands())

            if ch == self.img_dims[0]: # RGB
                d_img = img

            elif img.mode == 'RGBA':
                d_img = Image.new('RGB', img.size, color=(255,255,255))
                d_img.paste(img, mask=img.split()[3])

            else:
                print("Skipped", img_path, "dims were to small")
                batch_end = batch_end + 1

                if idx >= (batch_end-1):
                    break
                else:
                    continue # skip next part


            img_dspl = self.display(d_img)
            batch_dspl.append(img_dspl.permute(1, 2, 0))
            # get torch input
            img_pp = self.preprocess(d_img)
            img_tensor = img_pp.unsqueeze(0) # create a mini-batch as expected by the model
            batch = torch.vstack((batch, img_tensor))
            batch_paths.append(img_path.replace(".",""))


            if idx >= (batch_end-1):
                break

        return batch,batch_dspl,batch_paths

    def simBatch(self, input_batch, batch_size=1):
        '''
        sim a bigger batch size
        '''
        # hack to generate bigger batches from single image test function
        for div in range((batch_size - 1).bit_length()):
            input_batch = torch.cat((input_batch,input_batch))

        return input_batch

