"""
Author: Qizhi Li
预处理数据, 包括:
    1. 加载数据
    2. 切词
    3. 转换为词向量
"""
import os
import time
import pyprind
from gensim.models import KeyedVectors
from xml.dom.minidom import parse
import jieba
import re
import numpy as np
import matplotlib.pyplot as plt


def load_w2v():
    """
    加载word2vec
    :return:
    """
    start = time.time()
    print('Start loading w2v')
    file_name = 'cc.zh.300.vec'
    w2v = KeyedVectors.load_word2vec_format('./data/%s' % file_name, binary=False)

    print('w2v has loaded, took %.2f sec' % (time.time() - start))
    return w2v


def load_nlpcc_data(file_name):
    """
    获取nlpcc情感分析数据集
    :param file_name: str
            文件名
    :return X: list
            文本数据
    :return y: list
            标签数据
    """
    X = []  # 文本
    y = []  # 标签

    dom_tree = parse('./data/%s' % file_name)
    root_node = dom_tree.documentElement  # 获得根节点

    weibos = root_node.getElementsByTagName('weibo')  # 获得微博节点
    print('start loading xml file')
    pper = pyprind.ProgPercent(len(weibos), monitor=True)
    for weibo in weibos:
        sentences = weibo.getElementsByTagName('sentence')  # 获得句子节点
        for sentence in sentences:
            if sentence.hasAttribute('polarity'):
                X.append(sentence.childNodes[0].data)
                y.append(0 if sentence.getAttribute('polarity') == 'NEG' else 1)

        pper.update()

    return X, y


def clean_data(X):
    """
    清洗数据, 包括:
        1. 清理标点符号
        2. 切词
    :param X: list
            文本数据
    :return cleaned_X: list
            清洗后的文本数据
            [[word11, word12, ...], [word21, word22, ...], ...]
    """
    not_word = re.compile('[^A-Za-z0-9\u4e00-\u9fa5]')  # 非中文, 数字, 英文
    cleaned_X = []
    for sentence in X:
        cleaned_sentence = []
        cleaned_words = re.sub(not_word, ' ', sentence)  # 如果不用空格, 则断句会有问题, 影响jieba性能
        segment = jieba.cut(cleaned_words)
        for cleaned_word in segment:
            if cleaned_word is not ' ':
                cleaned_sentence.append(cleaned_word)

        cleaned_X.append(cleaned_sentence)

    return cleaned_X


def get_idx_word_mapping(cleaned_X):
    """
    获得id与词语之间的映射关系
    :param cleaned_X: list
            [[word11, word12, ...], [word21, word22, ...], ...]
            清洗后的文本数据
    :return idx2char: dict
            {1: 'word1', 2: 'word2', ...}
            id与词的映射
    :return char2idx: dict
            {'word1': 1, 'word2': 2, ...}
            词与id的映射
    :return word_set: set
            数据中的所有词
    """
    idx2char = {}
    char2idx = {}
    word_list = []
    # 获得全部词语
    for sentence in cleaned_X:
        for word in sentence:
            word_list.append(word)

    word_set = set(word_list)  # 去重

    for i, word in enumerate(word_set, start=1):
        idx2char[i] = word
        char2idx[word] = i
        
    return idx2char, char2idx, word_set


def get_unknown_word_embedding(w2v, word_set):
    """
    获得未知词的embedding
    :param w2v: Object
            词向量
    :param word_set: set
            数据中的所有词
    :return unknown_embedding: ndarray
            句子中所有出现过的词语的词向量的平均数
    """
    all_vectors = np.zeros((len(word_set), 300))
    # unknown_embedding = np.zeros((len(word_set), w2v[0]))

    count = 0
    for word in word_set:
        try:
            all_vectors[count] = w2v[word]
            count += 1
        except KeyError:
            count += 1
            continue

    # 删除all_vectors中的全0向量
    flag = (all_vectors == 0).all(1)  # 计算哪些行全0
    word_vectors = all_vectors[~flag, :]  # 删除全0元素

    unknown_embedding = np.mean(word_vectors, axis=0)

    return unknown_embedding


def get_sentence_length_distribution(cleaned_X):
    """
    获得句子的长度
    :param cleaned_X: list
            [[word11, word12, ...], [word21, word22, ...], ...]
            清洗后的文本数据
    """
    seq_length_frequency = {}

    for sentence in cleaned_X:
        seq_length = int(len(sentence) / 10) * 10  # 以10为间隔

        try:
            seq_length_frequency[seq_length] += 1
        except KeyError:
            seq_length_frequency[seq_length] = 1

    x = []  # 句子长度
    y = []  # 出现次数

    for length in seq_length_frequency.keys():
        x.append(length)
        y.append(seq_length_frequency[length])

    plt.xlabel('sequence length')
    plt.ylabel('The number of occurrences')
    plt.bar(x, y)
    plt.show()


def word_embedding(w2v, unknown_embedding, cleaned_X):
    """
    获得词嵌入
    :param w2v: Object
            word2vec向量
    :param unknown_embedding: ndarray
            shape: (300, )
            未知词向量
    :param cleaned_X: list
            [[word11, word12, ...], [word21, word22, ...], ...]
            清洗后的文本数据
    :return X_w2v: ndarray
            shape: (num_seq, max_seq_length, w2v_dim)
            词嵌入后的数据
    """
    max_seq_length = 30  # 句子中的最大词语数
    w2v_dim = 300  # 词向量维度
    num_seq = len(cleaned_X)  # 句子数

    X_w2v = np.zeros((num_seq, max_seq_length, w2v_dim))

    seq_count = 0
    for seq in cleaned_X:
        word_count = 0
        for word in seq:
            try:
                X_w2v[seq_count][word_count] = w2v[word]
            except KeyError:
                X_w2v[seq_count][word_count] = unknown_embedding
            word_count += 1

            # 如果词比max_seq_length多就跳过
            if word_count >= max_seq_length:
                break
        seq_count += 1

    return X_w2v


def preprocess():
    """
    预处理
    :return X_w2v: ndarray
            shape: (num_seq, max_seq_length, w2v_dim)
            词语的w2v嵌入
    :return labels: ndarray
            shape: (num_seq)
            标签
    """
    if os.path.exists('./data/X_w2v.npy'):
        X_w2v = np.load('./data/X_w2v.npy')
        labels = np.load('./data/labels.npy')
    else:
        w2v = load_w2v()
        file_name = 'ipad.xml'
        X, labels = load_nlpcc_data(file_name)
        cleaned_X = clean_data(X)
        idx2char, char2idx, word_set = get_idx_word_mapping(cleaned_X)
        unknown_embedding = get_unknown_word_embedding(w2v, word_set)
        # get_sentence_length_distribution(cleaned_X)
        X_w2v = word_embedding(w2v, unknown_embedding, cleaned_X)
        labels = np.array(labels)
        np.save('./data/X_w2v.npy', X_w2v)
        np.save('./data/labels.npy', labels)

    return X_w2v, labels









