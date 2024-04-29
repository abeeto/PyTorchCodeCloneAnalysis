import torch
from torch.utils.data import DataLoader

import csv, argparse

from model import TransE

TRAIN_DATASET_PATH = '/Users/sheep/research/datasets/FB15k/train.txt'
#TEST_DATASET_PATH = './dataset/FB15k/test.txt'
TEST_DATASET_PATH = '/Users/sheep/research/datasets/FB15k/test.txt'
GPU = torch.cuda.is_available()

def load(file_path):
    with open(file_path, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        data = []
        entity_dic = {}
        link_dic = {}
        entity_idx = 0
        link_idx = 0
        for row in tsv_reader:
            head = row[0]
            link = row[2]
            tail = row[1]
            if not head in entity_dic:
                entity_dic[head] = entity_idx
                entity_idx += 1

            if not tail in entity_dic:
                entity_dic[tail] = entity_idx
                entity_idx += 1

            if not link in link_dic:
                link_dic[link] = link_idx
                link_idx += 1
            data.append((entity_dic[head], link_dic[link], entity_dic[tail]))
        return data, entity_dic, link_dic

def load_with_dic(file_path, entity_dic, relation_dic):
    with open(file_path, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        data = []
        for row in tsv_reader:
            head = row[0]
            link = row[2]
            tail = row[1]
            data.append((entity_dic[head], relation_dic[link], entity_dic[tail]))
        return data


if __name__ == '__main__':
    device = torch.device("cuda" if GPU else "cpu")

    train_data, entity_dic, link_dic = load(TRAIN_DATASET_PATH)
    train_data_tensor = torch.tensor(train_data)
    test_data = load_with_dic(TEST_DATASET_PATH, entity_dic, link_dic)
    test_data_tensor = torch.tensor(test_data)

    transe = TransE(nentity=len(entity_dic), nrelation=len(link_dic), \
        vector_dim=50, margin=1., device=device)
    transe.fit(train_data_tensor, nepoch=1000, validation=test_data_tensor)
