import utils
import tqdm
import spacy
import tqdm
import options
import argparse
import data
import os
import numpy as np
from collections import Counter

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, word_counter, char_counter, tokenizer, process_text=None):
    source = utils.load_json(filename)
    total = 0
    examples = []
    process_text = process_text or (lambda text: text)

    for article in tqdm.tqdm(source['data']):
        for para in article['paragraphs']:
            context = process_text(para['context'].replace("''", '" ').replace("``", '" '))
            context_tokens = tokenizer(context)
            context_chars = [list(token) for token in context_tokens]
            spans = convert_idx(context, context_tokens)
            for token in context_tokens:
                word_counter[token] += len(para['qas'])
                for char in token:
                    char_counter[char] += len(para['qas'])
            for qa in para['qas']:
                total += 1
                ques = process_text(qa['question'].replace("''", '" ').replace("``", '" '))
                ques_tokens = tokenizer(ques)
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                y1s, y2s = [], []
                answer_texts = []
                for answer in qa['answers']:
                    answer_text = process_text(answer['text'])
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    if not answer_span:
                        print('{} in ({}, {}) out of context boundary'.format(answer_text, answer_start, answer_end))
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                example = {
                    'context_tokens': context_tokens,
                    'context_chars': context_chars,
                    'question_tokens': ques_tokens,
                    'question_chars': ques_chars,
                    'y1s': y1s,
                    'y2s': y2s,
                    'id': total
                }
                examples.append(example)
    return examples


def process_cmrc(filename, word_counter, char_counter, tokenizer):
    source = utils.load_json(filename)
    total = 0
    examples = []

    for para in tqdm.tqdm(source):
            context = para['context_text']
            context_tokens = tokenizer(context)
            context_chars = [list(token) for token in context_tokens]
            spans = convert_idx(context, context_tokens)
            for token in context_tokens:
                word_counter[token] += len(para['qas'])
                for char in token:
                    char_counter[char] += len(para['qas'])
            for qa in para['qas']:
                total += 1
                ques = qa['query_text']
                ques_tokens = tokenizer(ques)
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                y1s, y2s = [], []
                answer_texts = []
                for answer_text in qa['answers']:
                    answer_text = str(answer_text)
                    if answer_text.endswith('。'):
                        answer_text = answer_text[:-1]
                    try:
                        answer_start = context.index(answer_text)
                    except:
                        continue
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    if not answer_span:
                        print('{} in ({}, {}) out of context boundary'.format(answer_text, answer_start, answer_end))
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                if not y1s or not y2s:
                    continue
                example = {
                    'context_tokens': context_tokens,
                    'context_chars': context_chars,
                    'question_tokens': ques_tokens,
                    'question_chars': ques_chars,
                    'y1s': y1s,
                    'y2s': y2s,
                    'id': total
                }
                examples.append(example)
    return examples


def get_embedding(counter, limit=-1, emb_file=None, vec_size=None, token2idx_dict=None):
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None and os.path.isfile(emb_file):
        with open(emb_file, 'r', encoding='utf-8') as fh:
            for line in fh:
                array = line.split()
                word = ''.join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
    else:
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.01) for _ in range(vec_size)]

    token2idx_dict = token2idx_dict or {t:i for i,t in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[data.NULL] = data.NULL_ID
    token2idx_dict[data.OOV] = data.OOV_ID
    embedding_dict[data.NULL] = [0] * vec_size
    embedding_dict[data.OOV] = [0] * vec_size
    idx2emb_dict = {i:embedding_dict[t] for t,i in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[i] for i in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def make_options():
    parser = argparse.ArgumentParser()
    options.preprocess_opts(parser)
    options.data_opts(parser)
    return parser.parse_args()


if __name__ == '__main__':
    opts = make_options()
    word_counter, char_counter = Counter(), Counter()
    utils.rmdir('./generate')
    if opts.dataset == 'squad':
        train_examples = process_file(opts.squad_train_file, word_counter, char_counter, word_tokenize)
        dev_examples = process_file(opts.squad_dev_file, word_counter, char_counter, word_tokenize)
        test_examples = process_file(opts.squad_test_file, word_counter, char_counter, word_tokenize)
    elif opts.dataset in ['drcd', 'cmrc']:
        if opts.cws == 'ltp':
            import pyltp
            segmentor = pyltp.Segmentor()
            segmentor.load('./data/ltp_data/cws.model')
            def tokenize(text):
                return segmentor.segment(text)
        elif opts.cws == 'snownlp':
            from snownlp import SnowNLP
            def tokenize(text):
                s = SnowNLP(text)
                return s.words
        elif opts.cws == 'jieba':
            import jieba
            def tokenize(text):
                return list(jieba.cut(text))
        else:
            tokenize = list
        if opts.dataset == 'drcd':
            train_examples = process_file(opts.drcd_train_file, word_counter, char_counter, tokenize)
            dev_examples = process_file(opts.drcd_dev_file, word_counter, char_counter, tokenize)
            test_examples = process_file(opts.drcd_test_file, word_counter, char_counter, tokenize)
        elif opts.dataset == 'cmrc':
            train_examples = process_cmrc(opts.cmrc_train_file, word_counter, char_counter, tokenize)
            dev_examples = process_cmrc(opts.cmrc_dev_file, word_counter, char_counter, tokenize)
            test_examples = process_cmrc(opts.cmrc_test_file, word_counter, char_counter, tokenize)

    word_emb_mat, word2idx_dict = get_embedding(word_counter, emb_file=opts.glove_word_emb_file if opts.dataset == 'squad' else None, vec_size=opts.word_dim)
    char_emb_mat, char2idx_dict = get_embedding(char_counter, vec_size=opts.char_dim)
    assert len(word_emb_mat) == len(word2idx_dict)
    assert len(char_emb_mat) == len(char2idx_dict)

    utils.save_json(opts.train_example_file, train_examples)
    utils.save_json(opts.dev_example_file, dev_examples)
    utils.save_json(opts.test_example_file, test_examples)

    utils.save_json(opts.word_emb_file, word_emb_mat)
    utils.save_json(opts.char_emb_file, char_emb_mat)
    utils.save_json(opts.w2i_file, word2idx_dict)
    utils.save_json(opts.c2i_file, char2idx_dict)
    utils.save_json(opts.meta_file, {
        'dataset': opts.dataset,
        'num_train': len(train_examples),
        'num_dev': len(dev_examples),
        'num_test': len(test_examples),
        'word_vocab_size': len(word2idx_dict),
        'char_vocab_size': len(char2idx_dict)
    })
    print('vocab size: {}/{}'.format(len(word2idx_dict), len(char2idx_dict)))
    print('samples: {}/{}/{}'.format(len(train_examples), len(dev_examples), len(test_examples)))
    print('preprocess done.')