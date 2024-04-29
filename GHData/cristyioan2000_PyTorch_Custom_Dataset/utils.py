from PIL import Image
import numpy as np
from tqdm import tqdm


def data_names_create(root,img_flag = True, lbls_flag = False):
    '''
    :param root: path to the folder containing imgs/ lbls/
    :param img_flag: Generates data.names for imgs
    :param lbls_flag:Generates data.names for lbls
    :Effect: creates a data.names file containing all the images
    '''
    img_path = os.path.join(root,'imgs/')
    with open(os.path.join(root,'data.names'),'w') as wr:
        for file_name in tqdm(os.listdir(img_path)):
            wr.write(f"{os.path.join(img_path,file_name)}\n")
def file_to_PIL(file_name,genereate_lbls = True):
    file_name = file_name
    target = None
    if genereate_lbls:
        target = file_name.replace('/imgs/','/lbls/').replace('.jpg','.txt')
    img = Image.open(file_name)
    return img,target
# data_names_create(root = r'D:\datasets\custom_dataset_test',img_flag=True)