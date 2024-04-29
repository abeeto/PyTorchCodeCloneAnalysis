# -*- encoding:utf8 -*-
import numpy as np


def getDict(datas):
    mydict = dict()
    for data in datas:
        for word in data.lower().split():
            if word not in mydict:
                mydict[word] = len(mydict)
    # 最后将未知单词加入字典
    mydict['UNK'] = len(mydict)

    return mydict


def load_set(datas, mydict):
    sentences = []
    for sent in datas:
        words = sent.lower().split()
        index = []
        for word in words:
            if word in mydict:
                index.append(mydict[word])
            else:
                index.append(mydict['UNK'])
        sentences.append(np.array(index, dtype=int))

    return sentences


def padding_and_generate_mask(sentences, time_step):
    new_sentences = np.zeros([len(sentences), time_step], dtype=int)
    new_mask = np.zeros([len(sentences), time_step], dtype=int)
    for i, sent in enumerate(sentences):
        slen = len(sent)
        if slen <= time_step:
            new_sentences[i, 0:slen] = sent
            new_mask[i, slen - 1] = 1
        else:
            new_sentences[i, :] = sent[0:time_step]
            new_mask[i, time_step - 1] = 1
    return new_sentences, new_mask


def batch_iter(paded_sents, masks, batch_size):
    sents = np.array(paded_sents)
    masks = np.array(masks)
    data_size = len(sents)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_size)

        return_sents = sents[start_index:end_index]
        return_masks = masks[start_index:end_index]

        yield return_sents, return_masks
