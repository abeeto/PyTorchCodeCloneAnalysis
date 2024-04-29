import torch
import numpy as np
import torchwordemb


def vector_loader(text_field_words):
    path = 'word_embedding/glove.sentiment.conj.pretrained.txt'
    words_dict, vec = torchwordemb.load_word2vec_text(path)
    embed_size = vec.size(1)

    # match
    count_list2 = []
    count = 0
    dict_cat = []
    for word in text_field_words:
        if word in words_dict:
            count += 1
            dict_cat.append(words_dict[word])
        else:
            dict_cat.append([0.0] * embed_size)
            count += 1
            count_list2.append(count - 1)
    count_data = len(text_field_words) - len(count_list2)

    # modify zero
    sum = []
    for j in range(embed_size):
        sum_col = 0.0
        for i in range(len(dict_cat)):
            sum_col += dict_cat[i][j]
            sum_col = float(sum_col / count_data)
            sum_col = round(sum_col, 6)
        sum.append(sum_col)

    for i in range(len(count_list2)):
        dict_cat[count_list2[i]] = sum

    return dict_cat

def vector_loader_old(text_field_words):
    path = 'word_embedding/glove.sentiment.conj.pretrained.txt'

    words = []
    words_dict = {}
    file = open(path, 'rt', encoding='utf-8')
    lines = file.readlines()
    embed_size = 300

    for line in lines:
        line_split = line.split(' ')
        word = line_split[0]
        nums = line_split[1:]
        nums = [float(e) for e in nums]
        words.append(word)
        words_dict[word] = nums

    # match
    count_list2 = []
    count = 0
    dict_cat = []
    for word in text_field_words:
        if word in words_dict:
            count += 1
            dict_cat.append(words_dict[word])
        else:
            dict_cat.append([0.0] * embed_size)
            count += 1
            count_list2.append(count - 1)
    count_data = len(text_field_words) - len(count_list2)

    # modify zero
    sum = []
    for j in range(embed_size):
        sum_col = 0.0
        for i in range(len(dict_cat)):
            sum_col += dict_cat[i][j]
            sum_col = float(sum_col / count_data)
            sum_col = round(sum_col, 6)
        sum.append(sum_col)

    for i in range(len(count_list2)):
        dict_cat[count_list2[i]] = sum

    return dict_cat


def vector_loader_zero(text_field_words):
    # load word2vec_raw
    path = 'word_embedding/glove.6B.300d.txt'
    words = []
    words_dict = {}
    file = open(path, 'rt', encoding='utf-8')
    lines = file.readlines()
    embed_size = 300

    for line in lines:
        line_split = line.split(' ')
        word = line_split[0]
        nums = line_split[1:]
        nums = [float(e) for e in nums]
        words.append(word)
        words_dict[word] = nums

    # match
    count = 0
    dict_cat = []
    for word in text_field_words:
        if word in words_dict:
            count += 1
            dict_cat.append(words_dict[word])
        else:
            dict_cat.append([0.0] * embed_size)

    return dict_cat


def vector_loader_modify(text_field_words):
    path = 'word_embedding/glove.6B.300d.txt'
    words = []
    words_dict = {}
    file = open(path, 'rt', encoding='utf-8')
    lines = file.readlines()
    embed_size = 300

    for line in lines:
        line_split = line.split(' ')
        word = line_split[0]
        nums = line_split[1:]
        nums = [float(e) for e in nums]
        words.append(word)
        words_dict[word] = nums

    uniform = np.random.uniform(-0.1, 0.1, embed_size).round(6).tolist()     # uniform distribution U(a,b).均匀分布
    # match
    count_list2 = []
    count = 0
    dict_cat = []
    for word in text_field_words:
        if word in words_dict:
            count += 1
            dict_cat.append(words_dict[word])
        else:
            # a = torch.normal(mean=0.0, std=torch.arange(0.09, 0, -0.09))
            dict_cat.append(uniform)
            count += 1
            count_list2.append(count - 1)
    # count_data = len(text_field_words) - len(count_list2)

    return dict_cat

def vector_loader_rand(text_field_words):
    embed_size = 300
    # match
    text_words_size = len(text_field_words)
    dict_cat = torch.randn(text_words_size, embed_size)
    dict_cat = dict_cat.numpy()
    dict_cat = dict_cat.tolist()

    return dict_cat
