# -*- coding: utf-8 -*-

import os
from collections import Counter, OrderedDict as OD
from copy import copy

import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import string
import re

import torch
import torchvision as vis
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset
# from torchvision.transforms import transforms



def images_info(path):
    ## Get dataset images info
    jpg_files = os.listdir(path)
    # print(jpg_files)
    print("Number of photos in Flickr8k: %d" % (len(jpg_files)))
    return jpg_files


def annots_info(file, df=False):
    ## Get image caption text info
    with open(file, 'r') as ann_file:
        # print(ann_file)
        ann_text = ann_file.read()
        annots = []
        for line in ann_text.split('\n'):
            parts = line.split('\t')
            if len(parts) > 1:
                annots.append(parts[0].split('#') + [parts[-1].lower()])
            # print(annots[0])
    print("Number of annots in Flickr8k: %d" % (len(annots)))
    ## Dataframe annotations
    annots_df = pd.DataFrame(annots, columns=['filename', 'idx', 'caption'])
    ## Get unique filenames
    fnames_unq = np.unique(annots_df.filename.values)
    print("The number of unique filname are: %s" % len(fnames_unq))
    ## Captions per image
    cap_per_file = list(Counter(Counter(annots_df.filename.values).values()).keys())[0]
    print("The number of captions per image file: %d." % cap_per_file)
    if df:
        return annots_df


def tokenizer(text):
    # print(len(text))
    # return
    max_num_words = 8000
    tokenize = Tokenizer(num_words=max_num_words)
    tokenize.fit_on_texts(text)
    # print(tokenize)
    # print(tokenize.word_index)
    vocab_size = len(tokenize.word_index) + 1
    print("Vocab size: ", vocab_size)
    # return
    text2int = tokenize.texts_to_sequences(text)
    # print(text2int[:5])
    max_ln = np.max([len(cap) for cap in text2int])
    return [text2int, vocab_size, max_ln, tokenize]


def split_dset(data, nv, nt):
    evalnset = data[:nt]
    validset = data[nt:nt+nv]
    trainset = data[nt+nv:]
    return [trainset, validset, evalnset]


def prep_data(image_data, caption_data, vocab_size, cap_max_len):
    n = len(image_data)
    print("Number of images: %d" % n)
    assert(n == len(caption_data), "Number of images <--> captions should be equal.")

    images, captions, target_caps = [], [], []
    print("Max caption length: %d" % cap_max_len)
    for img, txt in zip(image_data, caption_data):
        for i in range(1, len(txt)):
            # print(txt[:i])
            in_txt = txt[:i]
            out_txt = txt[i]
            in_txt = torch.from_numpy(pad_sequences([in_txt], maxlen=cap_max_len).flatten())
            # print(txt, "Len in", len(in_txt), out_txt, sep='\n')
            # print(in_txt, out_txt, sep='\n')
            images.append(img.unsqueeze(0))
            captions.append(in_txt.unsqueeze(0))
            target_caps.append(torch.LongTensor([out_txt]))

    images = torch.cat(images)
    captions = torch.cat(captions).long()
    target_caps = torch.cat(target_caps)
    # print(images.dtype, captions.dtype, target_caps)
    return [images, captions, target_caps]



def read_image(dir_photos, normalize=False, resize=False, tensor=False):
    transform = []
    imgs = OD()
    if resize:
        transform.append(vis.transforms.Resize(256))
        transform.append(vis.transforms.CenterCrop(resize))
    if tensor:
        transform.append(vis.transforms.ToTensor())
    if normalize:
        transform.append(vis.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))
    transform = vis.transforms.Compose(transform)

    jpgs = images_info(dir_photos)
    for jpg in jpgs:
        img = Image.open(os.path.join(dir_photos, jpg))
        # print(transform(img).size())
        imgs[jpg] = transform(img)
    return imgs


def add_start_end_tokens(df):
    print("Adding start and end tokens ... ", end="")
    for i, cpt in enumerate(df.caption.values):
        df["caption"].iloc[i] = "startseq " + cpt + " endseq"
    print("done.")
    return df


def remove_punctuations(s):
    text_no_punctuation = s.translate(str.maketrans('','',string.punctuation))
    return (text_no_punctuation)


def remove_single_chars(s):
    text_len_more_than1 = ""
    for word in s.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return (text_len_more_than1)


def remove_numeric_vals(s, printTF=False):
    text_no_numeric = ""
    for word in s.split():
        isalpha = word.isalpha()
        if printTF:
            print("    {:10} : {:}".format(word, isalpha))
        if isalpha:
            text_no_numeric += " " + word
    return (text_no_numeric)


def clean_text(text):
    text = remove_punctuations(text)
    text = remove_single_chars(text)
    text = remove_numeric_vals(text)
    return text


def word_freq(df_text):
    # print(df_text)
    all_words = []
    for cpt in df_text.caption.values:
        all_words += cpt.split()
    print("Vocab size: %d" % len(set(all_words)))
    counts = Counter(all_words)
    # print(counts)
    df_counts = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_counts = df_counts.rename(columns={"index": "word", 0: "count"})
    df_counts = df_counts.sort_values(by=["count"], ascending=False)
    return df_counts.reset_index(drop=True)


import torch
import torch.utils.data
from torch.utils.data import Dataset


class Flickr8k(Dataset):

    def __init__(self, images, captions, target_caps):
        self.a, self.b, self.c = images, captions, target_caps

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        self.w = (self.a[idx], self.b[idx], self.c[idx])
        return self.w


def histogram(counts):
    # print(counts)
    # df_counts =
    plt.figure(figsize=(20, 3))
    plt.bar(counts.index, counts["count"])
    # plt.yticks(fontsize=20)
    # plt.xticks(counts.index, counts["word"], rotation=90, fontsize=20)
    plt.show()