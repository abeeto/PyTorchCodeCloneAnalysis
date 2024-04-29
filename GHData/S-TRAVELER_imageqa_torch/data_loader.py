import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np

import pdb
import random
import pickle as pkl
import os
import logging
import h5py
import scipy.sparse
import scipy.sparse as sparse
from collections import OrderedDict

logger = logging.getLogger('root')

#创建子类
class SanDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, data_folder, feature_file):
        self._image_feat = self.load_image_feat(data_folder, feature_file)
        question_id = OrderedDict()
        image_id = OrderedDict()
        question = OrderedDict()
        # answer set
        answer = OrderedDict()
        # answer counter
        answer_counter = OrderedDict()
        # most common answer
        answer_label = OrderedDict()
        splits = ['train', 'val1']
        
        for split in splits:
            with open(os.path.join(data_folder,split)+'.pkl','rb')as f:
                question_id[split] = pkl.load(f,encoding='iso-8859-1')
                image_id[split] = pkl.load(f,encoding='iso-8859-1')
                question[split] = pkl.load(f,encoding='iso-8859-1')
                answer[split] = pkl.load(f,encoding='iso-8859-1')
                answer_counter[split] = pkl.load(f,encoding='iso-8859-1')
                answer_label[split] = pkl.load(f,encoding='iso-8859-1')

       
        self._question_id = np.concatenate([question_id['train'],
                                                         question_id['val1']],
                                                        axis = 0)
        self._image_id = np.concatenate([image_id['train'],
                                                      image_id['val1']],
                                                     axis = 0)
        self._question = np.concatenate([question['train'],
                                                      question['val1']],
                                                     axis = 0)
        self._answer = np.concatenate([answer['train'],
                                                    answer['val1']],
                                                   axis = 0)
        self._answer_counter \
            = np.concatenate([answer_counter['train'],
                              answer_counter['val1']],
                             axis = 0)
        self._answer_label \
            = np.concatenate([answer_label['train'],
                              answer_label['val1']],
                             axis = 0)

        logger.info('finished loading data')

    def load_image_feat(self, data_path, h5_file):
        image_h5 = h5py.File(os.path.join(data_path, h5_file), 'r')
        shape = image_h5['shape']
        data = image_h5['data']
        col_idx = image_h5['indices']
        count_idx = image_h5['indptr']
        return scipy.sparse.csr_matrix((data, col_idx, count_idx),
                                       dtype='float32',
                                       shape=(shape[0], shape[1]))


    #返回数据集大小
    def __len__(self):
        return self._question.shape[0]

    #得到数据内容和标签
    def __getitem__(self, index):
        batch_image_id = self._image_id[index]
        batch_question = self._question[index]
        # index - 1 as query for image feature
        batch_image_feat = self._image_feat[batch_image_id]
        batch_image_feat = torch.Tensor(batch_image_feat.todense())

        batch_answer_label = torch.Tensor(self._answer_label[index])

        return batch_question, batch_image_feat, batch_answer_label
 
if __name__ == '__main__':
    dataset = SanDataset('D:\\documents\\code\\py\\imageqa-san\\data_vqa\\imageqa_san_data',  'trainval_feat.h5')
    print('dataset大小为：', dataset.__len__())
    print(dataset.__getitem__(0))
    print(dataset[0][0])
 
    #创建DataLoader迭代器
    dataloader = DataLoader.DataLoader(dataset,batch_size= 2, shuffle = False, num_workers= 4)
    for i, item in enumerate(dataloader):
        print('i:', i)
        data, label = item
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        print('data:', data[0])
        print('label:', label[0])