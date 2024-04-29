import os
from tools.txt_utils import *

for txt_name in ['train_aug.txt', 'train.txt', 'val.txt']:
    data_list = []

    for string in read_txt('./data/' + txt_name):
        image_id, _ = string.split(' ')
        image_id = os.path.basename(image_id).replace('.jpg', '')

        # print(image_id)
        data_list.append(image_id)

    write_txt('./data/' + txt_name, data_list)