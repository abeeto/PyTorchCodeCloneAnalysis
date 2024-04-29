import sys
import numpy
import numpy as np
from tempfile import TemporaryFile

import timeit
import cPickle
from conf import *

MENTION_TYPES = {
    "PRONOMINAL": 0,
    "NOMINAL": 1,
    "PROPER": 2,
    "LIST": 3
}
MENTION_NUM, SENTENCE_NUM, START_INDEX, END_INDEX, MENTION_TYPE, CONTAINED = 0, 1, 2, 3, 4, 5

DIR = args.DIR
embedding_file = DIR+"features/mention_data/word_vectors.npy"

span_dimention = 5*50
embedding_dimention = 50
embedding_size = 34275
word_embedding_dimention = 9*50

numpy.set_printoptions(threshold=numpy.nan)

def write_pair_feature_data(file_name):

    doc_path = DIR+"features/doc_data/%s/"%file_name
    pair_path = DIR+"features/mention_pair_data/%s/"%file_name
    mention_path = DIR+"features/mention_data/%s/"%file_name

    #embedding_matrix = numpy.load(embedding_file) 

    ## for mentions
    mention_spans = numpy.load(mention_path+"msp.npy")
    mention_word_index = numpy.load(mention_path+"mw.npy") 
    mention_feature = numpy.load(mention_path+"mf.npy")
    mention_id = numpy.load(mention_path+"mid.npy")[:,0]
    mention_did = numpy.load(mention_path+"mdid.npy")[:,0]
    mention_num = numpy.load(mention_path+"mnum.npy")[:,0]

    ## for pairs
    pair_feature = numpy.load(pair_path + 'pf.npy')
    pair_coref_info = numpy.load(pair_path + "y.npy")    
    pair_index = numpy.load(pair_path + "pi.npy")
    pair_mention_id = numpy.load(pair_path + "pmid.npy")

    ## for docs
    document_features = numpy.load(doc_path + 'df.npy')
    doc_pairs = numpy.load(doc_path + 'dpi.npy') # each line is the pair_start_index -- pair_end_index
    doc_mentions = numpy.load(doc_path + 'dmi.npy') # each line is the mention_start_index -- mention_end_index

    # build training data  
    doc_mention_sizes = {} # save the total mention size of each document

    pair_feature_list = []
    mention_feature_arrays = []
    
    for did in np.arange(doc_pairs.shape[0]):
        start_time = timeit.default_timer() 
        ps, pe = doc_pairs[did]
        ms, me = doc_mentions[did]

        doc_mention_sizes[did] = me - ms

        document_feature = document_features[did] 

        # build training data for each doc
        mention_span_indoc = mention_spans[ms:me]
        mention_word_index_indoc = mention_word_index[ms:me]
        mention_feature_indoc = mention_feature[ms:me]
        mention_num_indoc = mention_num[ms:me]
        
        mention_feature_list = []
        for mf in mention_feature_indoc:
            mention_feature_list.append(get_mention_features(mf,me-ms,document_feature))
        mention_feature_arrays += mention_feature_list
        mention_feature_list = numpy.array(mention_feature_list)
       
        pair_feature_indoc = pair_feature[ps:pe].astype(int)
        pair_coref_indoc = pair_coref_info[ps:pe].astype(int)

        ## generate training case
        pair_feature_index = 0

        for i in range(len(mention_feature_list)):
            ana_word_index = mention_word_index_indoc[i]
            ana_span = mention_span_indoc[i]
            ana_feature = mention_feature_list[i]
            
            candi_word_index = mention_word_index_indoc[:i]
            candi_span = mention_span_indoc[:i]
            
            pair_feature_array = []
            for j in range(i):
                #dis_feature = get_distance_features(mention_feature_list[i],mention_feature_list[j])
                dis_feature = get_distance_features(mention_feature_indoc[i],mention_feature_indoc[j])
                this_pair_feature = pair_feature_indoc[pair_feature_index] 
                pair_feature_index += 1
                #feature4this = numpy.concatenate((dis_feature,this_pair_feature,mention_feature_list[i],mention_feature_list[j]))
                #print this_pair_feature
                feature4this = numpy.concatenate((this_pair_feature,dis_feature,mention_feature_list[i][:-7],mention_feature_list[j]))
                #print len(feature4this),feature4this
            
                pair_feature_list.append(feature4this)
                #pair_feature_array.append(feature4this)
            #if len(pair_feature_array) == 0:
            #    pair_feature_array = numpy.zeros(77)
            #else:
            #    pair_feature_array = numpy.array(pair_feature_array)

            this_thrainig_data = (ana_word_index,ana_span,ana_feature,candi_word_index,candi_span,pair_feature_array)
            
            #pair_feature_list.append(pair_feature_array)

        end_time = timeit.default_timer()
        print >> sys.stderr, "Total use %.3f seconds for doc %d with %d mentions"%(end_time-start_time,did,me - ms)
    #print pair_feature_list
    pair_feature_list = numpy.array(pair_feature_list,dtype="float32")
    mention_feature_arrays = numpy.array(mention_feature_arrays,dtype="float32")

    numpy.save(mention_path+"yqy.npy",pair_feature_list)
    #numpy.save(mention_path+"mfyqy.npy",mention_feature_arrays)
    
def get_mention_features(mention_features, doc_mention_size,document_features):
    features = numpy.array([])
    features = numpy.append(features,one_hot(mention_features[MENTION_TYPE], 4)) 
    features = numpy.append(features,distance(np.subtract(mention_features[END_INDEX] - mention_features[START_INDEX], 1)))
    features = numpy.append(features,float(mention_features[MENTION_NUM])/ float(doc_mention_size))
    features = numpy.append(features,mention_features[CONTAINED])
    features = numpy.append(features,document_features)
    return features

def get_distance_features(m2, m1):
    dis_f = numpy.array(distance(m2[SENTENCE_NUM] - m1[SENTENCE_NUM]))
    dis_f = numpy.append(dis_f,distance(np.subtract(m2[MENTION_NUM] - m1[MENTION_NUM], 1)))
    dis_f = numpy.append(dis_f,int((m2[SENTENCE_NUM] == m1[SENTENCE_NUM]) & (m1[END_INDEX] > m2[START_INDEX])))
    #print m1[MENTION_NUM],m2[MENTION_NUM]
    #print np.subtract(m1[MENTION_NUM] - m2[MENTION_NUM], 1)
    return dis_f


def one_hot(a, n): 
    oh = np.zeros(n) 
    oh[a] = 1 
    return oh

def distance(a):
    d = np.zeros(11)
    d[a == 0, 0] = 1
    d[a == 1, 1] = 1
    d[a == 2, 2] = 1
    d[a == 3, 3] = 1
    d[a == 4, 4] = 1
    d[(5 <= a) & (a < 8), 5] = 1
    d[(8 <= a) & (a < 16), 6] = 1
    d[(16 <= a) & (a < 32), 7] = 1
    d[(a >= 32) & (a < 64), 8] = 1
    d[a >= 64, 9] = 1
    d[10] = np.clip(a, 0, 64) / 64.0
    return d

if __name__ == "__main__":
    #main()
    #write_pair_feature_data("test_reduced")
    write_pair_feature_data("dev_reduced")
    #write_pair_feature_data("train_reduced")

    #write_pair_feature_data("test")
    #write_pair_feature_data("train")
    #write_pair_feature_data("dev")
    print "Done!" 
