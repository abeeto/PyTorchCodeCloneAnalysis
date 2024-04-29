#coding=utf8
import sys
import numpy
import numpy as np
from tempfile import TemporaryFile
import random

import timeit
import cPickle
import json

from conf import *
import utils

MENTION_TYPES = {
    "PRONOMINAL": 0,
    "NOMINAL": 1,
    "PROPER": 2,
    "LIST": 3
}
MENTION_NUM, SENTENCE_NUM, START_INDEX, END_INDEX, MENTION_TYPE, CONTAINED = 0, 1, 2, 3, 4, 5

DIR = args.DIR
embedding_file = DIR+"features/mention_data/word_vectors.npy"

numpy.set_printoptions(threshold=numpy.nan)
random.seed(args.random_seed)

class DataGnerater():
    def __init__(self,file_name):

        doc_path = DIR+"features/doc_data/%s/"%file_name
        pair_path = DIR+"features/mention_pair_data/%s/"%file_name
        mention_path = DIR+"features/mention_data/%s/"%file_name

        gold_path = DIR+"gold/"+file_name.split("_")[0]
        # read gold chain
        self.gold_chain = {}
        gold_file = open(gold_path)
        golds = gold_file.readlines()
        for item in golds:
            gold = json.loads(item) 
            self.gold_chain[int(gold.keys()[0])] = list(gold[gold.keys()[0]])

    #embedding_matrix = numpy.load(embedding_file) 

    ## for mentions
        self.mention_spans = numpy.load(mention_path+"msp.npy")
        self.mention_word_index = numpy.load(mention_path+"mw.npy")[:, :-1] 
        self.mention_feature = numpy.load(mention_path+"mf.npy")
        self.mention_id = numpy.load(mention_path+"mid.npy")[:,0]
        self.mention_did = numpy.load(mention_path+"mdid.npy")[:,0]
        self.mention_num = numpy.load(mention_path+"mnum.npy")[:,0]

        self.mention_pair_feature = numpy.load(mention_path+"yqy.npy",mmap_mode='r')
        #self.mention_pair_feature = numpy.lib.format.open_memmap(mention_path+"pi.npy")

        #self.mention_feature_arrays = [] ## mention feature is saved in this array
        self.mention_feature_arrays = numpy.load(mention_path+"mfyqy.npy",mmap_mode='r')

    ## for pairs
        #self.pair_feature = numpy.load(pair_path + 'pf.npy')
        self.pair_coref_info = numpy.load(pair_path + "y.npy")    
        self.pair_index = numpy.load(pair_path + "pi.npy")
        self.pair_mention_id = numpy.load(pair_path + "pmid.npy")

    ## for docs
        self.document_features = numpy.load(doc_path + 'df.npy')
        self.doc_pairs = numpy.load(doc_path + 'dpi.npy') # each line is the pair_start_index -- pair_end_index
        self.doc_mentions = numpy.load(doc_path + 'dmi.npy') # each line is the mention_start_index -- mention_end_index

        self.batch = []
        self.doc_batch = {}
        # build training data  
        doc_index = range(self.doc_pairs.shape[0])
        done_num = 0
        total_num = self.doc_pairs.shape[0]
        estimate_time = 0.0
        self.n_anaphors = 0
        self.n_pairs = 0

        for did in doc_index:
            start_time = timeit.default_timer() 
            ps, pe = self.doc_pairs[did]
            ms, me = self.doc_mentions[did]
            self.n_anaphors += me - ms

            self.doc_batch[did] = []
            
            done_num += 1

            doc_mention_sizes = me - ms
            document_feature = self.document_features[did]

            #mention_feature_indoc = self.mention_feature[ms:me] 
            #for mf in mention_feature_indoc:
            #    self.mention_feature_arrays.append(get_mention_features(mf,me-ms,document_feature))

            max_pairs = 10000
            min_anaphor = 1
            min_pair = 0
            while min_anaphor < doc_mention_sizes:
                max_anaphor = min(new_max_anaphor(min_anaphor, max_pairs), me - ms)
                max_pair = min(max_anaphor * (max_anaphor - 1) / 2, pe - ps) 

                mentions = np.arange(ms, ms + max_anaphor)
                antecedents = np.arange(max_anaphor - 1)
                anaphors = np.arange(min_anaphor, max_anaphor)
                pairs = np.arange(ps + min_pair, ps + max_pair)
                pair_antecedents = np.concatenate([np.arange(ana)
                                                    for ana in range(min_anaphor, max_anaphor)]) 
                pair_anaphors = np.concatenate([(ana - min_anaphor) *
                                                    np.ones(ana, dtype='int32')
                                                    for ana in range(min_anaphor, max_anaphor)])
                '''
                print "mentions",len(mentions),mentions # 表示应该取的mention list
                print "ante",len(antecedents),antecedents #表示每个batch (mention list) 之中,要选取作为antecedents的index
                print "anaph",len(anaphors),anaphors #表示每个batch (mention list) 之中,要选取作为anaphors的index
                print "pairs",len(pairs),pairs #表示对应的pair_feature之类的信息
                print "pair_ante",len(pair_antecedents),pair_antecedentsi #表示要变成pair的antecedents index
                print "pair_ana",len(pair_anaphors),pair_anaphors #表示要变成pair的ana index
                '''


                positive, negative = [], []
                ana_to_pos, ana_to_neg = {}, {}

                ys = self.pair_coref_info[pairs]
                for i, (ana, y) in enumerate(zip(pair_anaphors, ys)):
                    labels = positive if y == 1 else negative
                    ana_to_ind = ana_to_pos if y == 1 else ana_to_neg
                    if ana not in ana_to_ind:
                        ana_to_ind[ana] = [len(labels), len(labels)]
                    else:
                        ana_to_ind[ana][1] = len(labels)
                    labels.append(i)

                # positive : index of positive examples in pairs
                # negative : index of negative examples in pairs

                #print len(positive),len(negative),"mentions",len(mentions),"pairs",len(pairs)
                #print "postive",positive
                #print "negative",negative

                pos_starts, pos_ends, neg_starts, neg_ends = [], [], [], []
                anaphoricities = []
                for ana in range(0, max_anaphor - min_anaphor):
                    if ana in ana_to_pos:
                        start, end = ana_to_pos[ana]
                        pos_starts.append(start)
                        pos_ends.append(end + 1)
                        anaphoricities.append(1)
                    else:
                        anaphoricities.append(0)
                    if ana in ana_to_neg:
                        start, end = ana_to_neg[ana]
                        neg_starts.append(start)
                        neg_ends.append(end + 1)
                # anaphoricities: = 1 if mention is anaphoricities else 0
                #print len(anaphoricities) = len(mentions),"pairs",len(pairs)
                #print anaphoricities

                starts, ends = [], []
                costs = []
                reindex = []
                pair_pos, anaphor_pos = 0, len(pairs)
                i, j = 0, 0
                for ana in range(0, max_anaphor - min_anaphor):
                    ana_labels = []
                    ana_reindex = []
                    start = i 
                    for ant in range(0, ana + min_anaphor):
                        ana_labels.append(ys[j])
                        i += 1
                        j += 1
                        ana_reindex.append(pair_pos)
                        pair_pos += 1
                    i += 1
                    ana_reindex.append(anaphor_pos)
                    anaphor_pos += 1

                    end = i 
                    ana_labels = np.array(ana_labels)
                    anaphoric = ana_labels.sum() > 0 
                    if end > (start + 1):
                        starts.append(start)
                        ends.append(end)
                        reindex += ana_reindex
                    else:
                        i = start
                        continue 

                    WL = 1.0
                    FL = 0.4
                    FN = 0.8

                    if anaphoric:
                        ana_costs = np.append(WL * (ana_labels ^ 1), FN)
                    else:
                        ana_costs = np.append(FL * np.ones_like(ana_labels), 0)
                    costs += list(ana_costs)

                positive = numpy.array(positive,dtype='int32')
                negative = numpy.array(negative,dtype='int32')
                pos_starts = np.array(pos_starts, dtype='int32')
                pos_ends = np.array(pos_ends, dtype='int32')
                neg_starts = np.array(neg_starts, dtype='int32')
                neg_ends = np.array(neg_ends, dtype='int32')
                reindex = np.array(reindex, dtype='int32')
                costs = np.array([costs], dtype='float')

                #print "score_index",np.concatenate([positive, negative])[:, np.newaxis]
                #print "starts",np.concatenate([pos_starts, positive.size + neg_starts])[:, np.newaxis]
                #print "ends",np.concatenate([pos_ends, positive.size + neg_ends])[:, np.newaxis]
                #print "y",np.concatenate([np.ones(pos_starts.size),np.zeros(neg_starts.size)])[:, np.newaxis]

                top = {}
                top["score_index"] = np.concatenate([positive, negative])
                top["starts"] = np.concatenate([pos_starts, positive.size + neg_starts])
                top["ends"] = np.concatenate([pos_ends, positive.size + neg_ends])
                top["top_gold"] = np.concatenate([np.ones(pos_starts.size),np.zeros(neg_starts.size)])

                rl = {}
                rl["starts"] = numpy.array(starts,dtype='int32')
                rl["ends"] = numpy.array(ends,dtype='int32')
                rl["did"] = did
                rl["reindex"] = reindex
                rl["costs"] = costs

                self.batch.append( (mentions,antecedents,anaphors,
                            pairs,pair_antecedents,pair_anaphors,
                            numpy.array(anaphoricities),positive,negative,top,rl) )

                self.doc_batch[did].append( (mentions,antecedents,anaphors,
                            pairs,pair_antecedents,pair_anaphors,
                            numpy.array(anaphoricities),positive,negative,top,rl) )


                min_anaphor = max_anaphor
                min_pair = max_pair
                self.n_pairs += len(pairs)
        
            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            #print >> sys.stderr, "Total use %.3f seconds for doc %d with %d mentions (%d/%d) -- EST:%f , Left:%f"%(end_time-start_time,did,me - ms,done_num,total_num,EST,EST-estimate_time)
        self.anaphors_per_batch = float(self.n_anaphors) / float(len(self.batch))
        self.pairs_per_batch = float(self.n_pairs)/float(len(self.batch))
        #self.mention_feature_arrays = numpy.array(self.mention_feature_arrays)

        self.scale_factor = self.pairs_per_batch
        #self.scale_factor = self.anaphors_per_batch
        self.anaphoricity_scale_factor = 50 * self.anaphors_per_batch 

        self.scale_factor_top = 10*self.anaphors_per_batch
        self.anaphoricity_scale_factor_top = 20 * self.anaphors_per_batch 

    def generater(self,shuffle=False):

        # build training data  
        doc_index = range(self.doc_pairs.shape[0])
        if shuffle:
            numpy.random.shuffle(doc_index) 

        batch = []

        done_num = 0
        total_num = self.doc_pairs.shape[0]
        estimate_time = 0.0
        for did in doc_index:
            start_time = timeit.default_timer() 
            ps, pe = self.doc_pairs[did]
            ms, me = self.doc_mentions[did]
            
            done_num += 1

            doc_mention_sizes = me - ms

            document_feature = self.document_features[did] 

            # build training data for each doc
            mention_span_indoc = self.mention_spans[ms:me]
            mention_word_index_indoc = self.mention_word_index[ms:me]
            #mention_feature_indoc = self.mention_feature[ms:me]
            mention_num_indoc = self.mention_num[ms:me]
            mention_id_real = self.mention_id[ms:me]

            mention_feature_list = self.mention_feature_arrays[ms:me]
        
            mention_pair_feature_indoc = self.mention_pair_feature[ps:pe]    

            pair_coref_indoc = self.pair_coref_info[ps:pe].astype(int)
            pair_mention_id_indoc = self.pair_mention_id[ps:pe]

            target_for_each_mention = []
            mention_id_for_each_mention = []
            pair_feature_for_each_mention = []
            st = 0
            for i in range(len(mention_feature_list)):
                target = pair_coref_indoc[st:st+i] # if mention is index r, it has r antecedents
                target_for_each_mention.append(target)

                pair_feature_current_mention = mention_pair_feature_indoc[st:st+i]
                pair_feature_for_each_mention.append(pair_feature_current_mention)

                mention_ids = pair_mention_id_indoc[st:st+i]
                
                this_mention_id = mention_id_real[i]
                candidates_id = [] 
                if len(mention_ids) > 0:
                    candidates_id = mention_ids[:,1].tolist()
                mention_id_for_each_mention.append((did,this_mention_id,candidates_id))
                st = st+i
           
            inside_index = range(len(mention_feature_list))
            if shuffle:
                numpy.random.shuffle(inside_index)

            for i in inside_index:
                ana_word_index = mention_word_index_indoc[i]
                ana_span = mention_span_indoc[i]
                ana_feature = mention_feature_list[i]
            
                candi_word_index = mention_word_index_indoc[:i]
                candi_span = mention_span_indoc[:i]

                pair_feature_array = pair_feature_for_each_mention[i]
             
                this_thrainig_data = (ana_word_index,ana_span,ana_feature,candi_word_index,candi_span,pair_feature_array,target_for_each_mention[i],mention_id_for_each_mention[i])
                ## mention_id_for_each_mention: list, each item is like : (doc_id,current_mention_id, candidate_id)
                #if len(pair_feature_array) > 0:
                #    print len(ana_word_index),len(ana_span),len(ana_feature),len(pair_feature_array[-1])

                doc_end = False
                if i == inside_index[-1]:
                    doc_end = True

                yield this_thrainig_data,doc_end

            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            print >> sys.stderr, "Total use %.3f seconds for doc %d with %d mentions (%d/%d) -- EST:%f , Left:%f"%(end_time-start_time,did,me - ms,done_num,total_num,EST,EST-estimate_time)

    def train_generater(self,filter_num=700,batch_size=10000,shuffle=False,top=False,reinforce=False):

        if shuffle:
            numpy.random.shuffle(self.batch) 

        done_num = 0
        total_num = len(self.batch)
        estimate_time = 0.0
        for mentions,antecedents,anaphors,pairs,pair_antecedents,pair_anaphors,anaphoricities,positive,negative,top_x,rl in self.batch:
            start_time = timeit.default_timer() 
            done_num += 1
            candi_word_index_return = self.mention_word_index[mentions[0]:mentions[-1]+1][antecedents]
            candi_span_return = self.mention_spans[mentions[0]:mentions[-1]+1][antecedents]
            candi_ids_return = self.mention_id[mentions[0]:mentions[-1]+1][antecedents]

            mention_word_index_return = self.mention_word_index[mentions[0]:mentions[-1]+1][anaphors]
            mention_span_return = self.mention_spans[mentions[0]:mentions[-1]+1][anaphors]

            pair_features_return = self.mention_pair_feature[pairs[0]:pairs[-1]+1]
            pair_target_return = self.pair_coref_info[pairs[0]:pairs[-1]+1].astype(int)

            anaphoricity_word_index = mention_word_index_return
            anaphoricity_span = mention_span_return
            anaphoricity_target = anaphoricities
            anaphoricity_feature = self.mention_feature_arrays[mentions[0]:mentions[-1]+1][anaphors]

            assert len(anaphoricities) == len(anaphoricity_feature)
            assert len(anaphoricities) == len(anaphoricity_span)

            if top == True:
                yield mention_word_index_return, mention_span_return, candi_word_index_return,candi_span_return,\
                pair_features_return,pair_antecedents,pair_anaphors,pair_target_return,positive,negative,\
                anaphoricity_word_index, anaphoricity_span, anaphoricity_feature, anaphoricity_target,top_x
            elif reinforce == True:
                yield mention_word_index_return, mention_span_return, candi_word_index_return,candi_span_return,\
                pair_features_return,pair_antecedents,pair_anaphors,pair_target_return,positive,negative,\
                anaphoricity_word_index, anaphoricity_span, anaphoricity_feature, anaphoricity_target,rl,candi_ids_return
            else:
                yield mention_word_index_return, mention_span_return, candi_word_index_return,candi_span_return,\
                pair_features_return,pair_antecedents,pair_anaphors,pair_target_return,positive,negative,\
                anaphoricity_word_index, anaphoricity_span, anaphoricity_feature, anaphoricity_target
            
            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            print >> sys.stderr, "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)

    def rl_case_generater(self,shuffle=False):

        index_list = range(len(self.doc_batch.keys()))

        if shuffle:
            random.shuffle(index_list) 

        done_num = 0
        total_num = len(self.doc_batch)
        estimate_time = 0.0

        for did_index in index_list:
            start_time = timeit.default_timer() 
            did = self.doc_batch.keys()[did_index]
            done_num += 1
            
            i = 0
            for mentions,antecedents,anaphors,pairs,pair_antecedents,pair_anaphors,anaphoricities,positive,negative,top_x,rl in self.doc_batch[did]:
                i += 1

                candi_word_index_return = self.mention_word_index[mentions[0]:mentions[-1]+1][antecedents]
                candi_span_return = self.mention_spans[mentions[0]:mentions[-1]+1][antecedents]
                candi_ids_return = self.mention_id[mentions[0]:mentions[-1]+1]
            
                mention_word_index_return = self.mention_word_index[mentions[0]:mentions[-1]+1][anaphors]
                mention_span_return = self.mention_spans[mentions[0]:mentions[-1]+1][anaphors]

                pair_features_return = self.mention_pair_feature[pairs[0]:pairs[-1]+1]
                pair_target_return = self.pair_coref_info[pairs[0]:pairs[-1]+1].astype(int)


                anaphoricity_word_index = mention_word_index_return
                anaphoricity_span = mention_span_return
                anaphoricity_target = anaphoricities
                anaphoricity_feature = self.mention_feature_arrays[mentions[0]:mentions[-1]+1][anaphors]

                assert len(anaphoricities) == len(anaphoricity_feature)
                assert len(anaphoricities) == len(anaphoricity_span)

                rl["end"] = False
                if i == len(self.doc_batch[did]):
                    rl["end"] = True

                yield mention_word_index_return, mention_span_return, candi_word_index_return,candi_span_return,\
                pair_features_return,pair_antecedents,pair_anaphors,pair_target_return,positive,negative,\
                anaphoricity_word_index, anaphoricity_span, anaphoricity_feature, anaphoricity_target,rl,candi_ids_return
            
            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            print >> sys.stderr, "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)

def new_max_anaphor(n, k): 
    # find m such that sum from i=n to m-1 is < k
    # i.e., total number of pairs with anaphor num between n and m (exclusive) < k
    return max(1, int(np.floor(0.5 * (1 + np.sqrt(8 * k + 4 * n * n - 4 * n + 1))))) 
    
def get_mention_features(mention_features, doc_mention_size,document_features):
    features = numpy.array([])
    features = numpy.append(features,one_hot(mention_features[MENTION_TYPE], 4)) 
    features = numpy.append(features,distance(np.subtract(mention_features[END_INDEX] - mention_features[START_INDEX], 1)))
    features = numpy.append(features,float(mention_features[MENTION_NUM])/ float(doc_mention_size))
    features = numpy.append(features,mention_features[CONTAINED])
    features = numpy.append(features,document_features)
    return features

def get_distance_features(m1, m2):
    dis_f = numpy.array(int((m2[SENTENCE_NUM] == m1[SENTENCE_NUM]) & (m1[END_INDEX] > m2[START_INDEX])))
    dis_f = numpy.append(dis_f,distance(m2[SENTENCE_NUM] - m1[SENTENCE_NUM]))
    dis_f = numpy.append(dis_f,distance(np.subtract(m2[MENTION_NUM] - m1[MENTION_NUM], 1)))
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
    #data = DataGnerater("test_reduced")   
    data = DataGnerater("dev_reduced")   
    data.train_generater()
    for t in data.train_generater():
        mention_word_index_return, mention_span_return, candi_word_index_return,candi_span_return,pair_features_return,pair_antecedents,pair_anaphors,target,positive,negative,anaphoricity_word_index, anaphoricity_span, anaphoricity_feature, anaphoricity_target = t
        pass
        #print pair_anaphors
    #for t in data.train_generater():
    #    candi_word_index, candi_span, mention_word_index, mention_span, feature_pair, target = t
    #    print len(candi_word_index),len(candi_span),len(mention_word_index),len(mention_span),len(feature_pair),len(target)
