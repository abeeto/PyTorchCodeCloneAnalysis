import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import re
root_dir = '/home/deep/cnn-text-classification-pytorch/reviews/'


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def read_dataset(domain):
    #files = ['positive.tsv', 'negative.tsv']
    reviews = []
    file_p = root_dir + domain + '/'

    with open(os.path.join(file_p, 'positive.tsv')) as f:
        for line in f.readlines():
            reviews.append([clean_str(line.split('\t')[0]).encode('utf-8'), 'positive'])

    with open(os.path.join(file_p, 'negative.tsv')) as f:
        for line in f.readlines():
            reviews.append([clean_str(line.split('\t')[0]).encode('utf-8'), 'negative'])

    reviews = np.array(reviews)
    print reviews[:, 1]
    train_x, train_y = reviews[:, 0], reviews[:, 1]
    #print train_x
    #train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, stratify=train_y,test_size=0.166, random_state=666)
    #print train_y
    return train_x, train_y


def write_train_val(datasets, training_f, val_f):
    training_f = open(training_f, 'w')
    train_x, val_x, train_y, val_y = datasets
    print len(train_x), len(val_x)
    train_dat = pd.DataFrame(train_x, columns=['review'])
    train_dat['labels'] = train_y
    train_dat.to_csv(training_f, index=False)
    #for i, review in enumerate(train_x):
    #    training_f.write(review.encode('utf-8') + '\t' + train_y[i] + '\n')
    training_f.close()
    #rint training_f
    val_dat = pd.DataFrame(val_x, columns=['review'])
    val_dat['labels'] = val_y
    val_dat.to_csv(val_f, index=False)


def write_reviews(datasets, test_f):
    training_f = open(test_f, 'w')
    test_x, test_y = datasets
    test_dat = pd.DataFrame(test_x, columns=['review'])
    test_dat['labels'] = test_y
    test_dat.to_csv(test_f, index=False)


def append_labels(out_list, dataset):
    for d in dataset:
        out_list.append(d)

    return out_list


if __name__ == '__main__':
    #out_dat = read_dataset('dvd')
    domains = ['kitchen', 'dvd', 'electronics', 'books']


    for i, domain in enumerate(domains):
        tr_x = []
        vl_x = []
        tr_y = []
        vl_y = []
        test_domain = domain
        train_domains = [dom for dom in domains if dom != test_domain]
        out_dir = 'leave_out_'+ test_domain
        if not os.path.exists(root_dir + out_dir):
            os.mkdir(root_dir + out_dir)
        for dom in train_domains:
            review, label = read_dataset(dom)

            train_x, val_x, train_y, val_y = train_test_split(review, label, stratify=label, test_size=0.167,
                                                          random_state=666)
            print len(train_x), len(val_x)
            tr_x, vl_x, tr_y, vl_y = append_labels(tr_x, train_x), append_labels(vl_x, val_x), append_labels(tr_y, train_y), \
                                     append_labels(vl_y, val_y)


        out_dat = tr_x, vl_x, tr_y, vl_y
        print len(tr_y), len(vl_x)
        write_train_val(out_dat, root_dir + out_dir + '/' + 'train.csv', root_dir + out_dir + '/' + 'val.csv')

        test_dat= read_dataset(test_domain)
        write_reviews(test_dat, root_dir + out_dir + '/' + 'test.csv')

    #write_train_val(out_dat, '/home/deep/cnn-text-classification-pytorch/reviews/training_samples/train.tsv'
    #                , '/home/deep/cnn-text-classification-pytorch/reviews/training_samples/val.tsv')
    print Counter(tr_y)
