import pandas as pd
import numpy as np
from gensim import models
import gensim
from gensim import corpora
import os

#review_d = '/home/deep/cnn-text-classification-pytorch/topic_lstm_torch/reviews/'
review_d = '/home/DebanjanChaudhuri/topic_lstm_torch/reviews'


def create_lda(train_path):
    print train_path.split('_')[-1]
    train_dat = np.array(pd.read_csv(train_path + '/train.csv')['review'])
    train_dat = [sent.split() for sent in train_dat]
    print train_dat[0]
    dictionary = corpora.Dictionary(train_dat)
    dictionary.save(train_path + '/lda_model/dict_' + train_path.split('_')[-1])
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in train_dat]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=50, id2word=dictionary, passes=50)
    print(ldamodel.print_topics(num_topics=10))
    ldamodel.save(train_path + '/lda_model/lda_' + train_path.split('_')[-1])


if __name__ == '__main__':
    for dir in os.listdir(review_d):
        create_lda(review_d + '/' + dir)