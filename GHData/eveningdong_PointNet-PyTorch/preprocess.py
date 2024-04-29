import os
import numpy as np
import glob
import pandas as pd
import time
import yaml
from os import makedirs
from os.path import exists, join


def shuffle_data(data, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def sample_data_label(data, label, num_sample):
    N = data.shape[0]
    if N < num_sample:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        new_data = np.concatenate([data, dup_data], 0)
        new_label = label[range(N) + list(sample)]
    else:
        new_data = data
        new_label = label
    return new_data, new_label


def process(file_name, config_name):
    df = pd.read_csv(file_name)
    data = df.values
    print('data.shape: ', data.shape)
    label = data[:, 6].astype(np.uint8)
    label = remap(label, config_name)
    data = np.delete(data, -1, axis=1)
    data[:, 3:6] /= 255.0
    data[:, 0] -= np.min(data[:, 0])
    data[:, 1] -= np.min(data[:, 1])
    data[:, 2] -= np.min(data[:, 2])

    make_blocks(data, label, file_name, NUM_POINT, block_size_input, stride_input,
                random_sample=False, sample_num=None, sample_aug=1)

def make_blocks(data, label, file_name, num_point, block_size, stride,
                random_sample=False, sample_num=None, sample_aug=1):
    limit = np.amax(data, 0)[0:3]
    
    # Get the corner location for our sampling blocks
    x_list = []
    y_list = []

    num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
    for i in range(num_block_x):
        for j in range(num_block_y):
            x_list.append(i * stride)
            y_list.append(j * stride)

    # Collect blocks
    for idx in range(len(x_list)):
        x = x_list[idx]
        y = y_list[idx]
        
        xcond = (data[:, 0] <= x + block_size) & (data[:, 0] >= x)
        ycond = (data[:, 1] <= y + block_size) & (data[:, 1] >= y)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
            continue

        block_data = data[cond, :]
        block_label = label[cond]

        # print(block_data.shape)
        # print(block_label.shape)

        # # randomly subsample data
        # block_data_sampled, block_label_sampled = \
        #     sample_data_label(block_data, block_label, num_point)

        minx = min(block_data[:, 0])
        miny = min(block_data[:, 1])
        block_data[:, 0] -= (minx + block_size / 2)
        block_data[:, 1] -= (miny + block_size / 2)

        block_data = block_data.astype(np.float32)
        file_name_split = file_name.split('/')[-1][:-4]

        data_store_name = output_dir + '/' + file_name_split + '_block_' + str(idx) + '_data.npy'
        label_store_name = output_dir + '/' + file_name_split + '_block_' + str(idx) + '_label.npy'

        np.save(data_store_name, block_data)
        np.save(label_store_name, block_label)
        print(data_store_name + ' finished')
    
def remap(label, file):
    data_config = yaml.safe_load(open(file, 'r'))
    remap_dict = data_config["remap_dict"]
    new_map = np.zeros(len(remap_dict.keys()), dtype=np.int32)
    new_map[list(remap_dict.keys())] = list(remap_dict.values())
    remapped_label = new_map[label]
    return remapped_label


if __name__ == "__main__":
    start = time.time()
    MODE = 'train'
    NUM_POINT = 4096 * 16

    base_dir = './data'
    data_dir = join(base_dir, MODE)
    output_dir = join(base_dir, 'processed_' + MODE)
    makedirs(output_dir) if not exists(output_dir) else None
    config_name = './sensat.yaml'

    block_size_input = float(np.sqrt(NUM_POINT / 4096))
    if MODE is 'train':
        stride_input = block_size_input / 2.0
    else:
        stride_input = block_size_input

    for file_name in glob.glob(join(data_dir, '*.csv')):
        process(file_name, config_name)
    print('total time is:' + str(time.time() - start))