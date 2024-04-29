import time
import sys
import logging
import numpy as np
import os
import re
import random
import datetime
from collections import Counter
import pandas as pd
UNK = "$UNK$"
NUM = "$NUM$"
label_UNK = "O"


def max_words_length(filepath):
    with open(filepath, 'r', encoding="utf-8-sig") as f:
        max_len, lines = 0, f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) < 1:
                continue
            max_len = max(len(line.split()[0]), max_len)
    return max_len


def shuffle_data(filename):
    with open(filename, 'r', encoding="utf-8-sig") as f:
        lines = f.readlines()
        S, collect = '', []
        for i, line in enumerate(lines):
            l = line.strip()
            if len(l) > 0:
                S += l + '\n'
            else:
                S += '\n'
                collect += S,
                S = ''
    random.shuffle(collect)
    with open(filename, 'w', encoding="utf-8-sig") as f:
        for sent in collect:
            f.write(sent)
    print("finish shuffle", os.path.split(filename)[-1])
    return filename


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(
            sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max(
            [max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(
            sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split("-")[0]
    tag_type = tag_name.split("-")[-1]
    return tag_class, tag_type


def get_chunks(seq, labels, DEFAULT):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        labels: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        labels = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = labels[DEFAULT]
    idx_to_tag = {idx: tag for tag, idx in labels.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks += chunk,
            chunk_type, chunk_start = None, None
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks += chunk,
                chunk_type, chunk_start = tok_chunk_type, i
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks += chunk,
    return chunks


def get_processing_word(vocab_words=None,
                        vocab_chars=None,
                        lowercase=False,
                        chars=False,
                        label_vocab=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """

    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            elif label_vocab is False:
                word = vocab_words[UNK]
            else:
                word = vocab_words[label_UNK]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


class Dataset(object):
    """
    Class that iterates over import Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = Dataset(filename)
        for sentence, tags in data:
            pass
        ```
    """

    def __init__(self,
                 filename,
                 processing_word=None,
                 processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        niter = 0
        with open(self.filename, 'r', encoding='utf-8-sig') as f:
            words, tags = [], []
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if len(line) == 0:
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    line_split = line.split()
                    word, tag = line_split[0], line_split[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length


def get_vocabs(datasets):
    data_name = ['train', 'dev', 'test']
    vocab_words, vocab_labels = set(), set()
    for i, dateset in enumerate(datasets):
        for words, labels in dateset:
            vocab_words.update(words)
            vocab_labels.update(labels)
        print("dataset {} - done, vocab update to {} tokens".format(
            data_name[i], len(vocab_words)))
    return vocab_words, vocab_labels


def get_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building glove vocab...")
    vocab = set()
    with open(filename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            word = line.strip().split(" ")[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab... -->", filename)
    with open(filename, "w", encoding='utf-8-sig') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = {}
        with open(filename, 'r', encoding='utf-8-sig') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
    except IOError as e:
        print("IOError:", e, ', there is no {} exist.'.format(filename))
    return d


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError as e:
        print("IOError:", e, ', there is no {} exist.'.format(filename))


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


# def get_default_labels(dataset, default_path):
#     nums = [label for _, label in dataset]
#     label_counts = Counter(nums)
#     res = label_counts.most_common(1)[0][0]
#     with open(default_path, 'w', encoding="utf-8-sig") as f:
#         f.write(res)
#     return res


def get_char_vocab(dataset):
    """
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """

    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def build_data(config):
    processing_word = get_processing_word(lowercase=True)

    # clean data
    train_filepath, dev_filepath_a = write_clear_data_pd(
        config.train_filename,
        config.DEFAULT,
        domain=config.domain,
        build_dev=config.build_dev_from_trainset,
        dev_ratio=config.dev_ratio)
    test_filepath, dev_filepath_b = write_clear_data_pd(
        config.test_filename,
        config.DEFAULT,
        domain=config.domain,
        build_dev=config.build_dev_from_testset,
        dev_ratio=config.dev_ratio)
    if (dev_filepath_a or dev_filepath_b) is None:
        dev_filepath, _ = write_clear_data_pd(
            config.dev_filename, config.DEFAULT, domain=config.domain)
    else:
        dev_filepath = (dev_filepath_a or dev_filepath_b)

    # train_filepath = clear_data_path(config.train_filename)
    # test_filepath = clear_data_path(config.test_filename)
    # dev_filepath = clear_data_path(config.dev_filename)

    shuffle_data(train_filepath)
    # shuffle_data(test_filepath)
    # shuffle_data(dev_filepath)

    # Generators
    dev = Dataset(dev_filepath, processing_word)
    test = Dataset(test_filepath, processing_word)
    train = Dataset(train_filepath, processing_word)

    # Build Word and labels vocab
    vocab_words, vocab_labels = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.glove_filename)

    # combine vocabs
    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.words_filename)
    write_vocab(vocab_labels, config.labels_filename)

    # Trim GloVe Vectors
    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename,
                                 config.trimmed_filename, config.dim)

    # Build and save char vocab
    train = Dataset(train_filepath)
    vocab_chars = get_char_vocab(train)
    # config.DEFAULT = get_default_labels(train,config.default_filename)
    write_vocab(vocab_chars, config.chars_filename)


def write_clear_data(from_path,
                     DEFAULT="O",
                     build_dev=False,
                     dev_ratio=0.3,
                     BIO=False):
    """
    Given adequate stv data file, clean out the data that we need for sloting tagging
    Args:
        from_path: (string) path of original stv file
        build_dev: (bool) do you need to build dev from data file
        dev_ratio: (float) ratio of data build from data file
    Returns:
        (sting) path of clear data file
    """
    from_file = open(from_path, 'r', encoding='utf-8-sig')
    to_path = os.path.splitext(from_path)[0] + '.txt'
    to_file = open(to_path, 'w', encoding='utf-8-sig')
    lines = from_file.readlines()
    data_name = os.path.split(to_path)[1]
    print('begin clean dataset to {}...'.format(data_name))
    if build_dev:
        dev_path = os.path.join(
            os.path.split(to_path)[0], 'dev' + os.path.splitext(to_path)[1])
        dev_file = open(dev_path, 'w', encoding='utf-8-sig')
        print('meanwhile generate clear dev data from {}...'.format(data_name))
    for idx, line in enumerate(lines):
        row = line.split('\t')
        label_chunk, S = False, []
        for string in row[1].split():
            m = re.match(r'^(</|<)([^<>]+)>$', string)
            if m is not None:
                label_chunk = not label_chunk
                label = m.group(2).lower()
                if not label_chunk:
                    if len(S) > 1:
                        for i in range(len(S)):
                            if i == 0:
                                S[i] = S[i][0] + "\tB-" + S[i][1] if BIO else S[i][0] + "\t" + S[i][1]
                            else:
                                S[i] = S[i][0] + "\tI-" + S[i][1] if BIO else S[i][0] + "\t" + S[i][1]
                    elif len(S) == 1:
                        S[0] = S[0][0] + "\t" + S[0][1]
                    to_file.write(''.join(S))
                    if build_dev and idx % (1 // dev_ratio) == 0:
                        dev_file.write(''.join(S))
                    S = []
            elif label_chunk:
                S += (string, label + '\n'),
            else:
                to_file.write(string + '\t' + DEFAULT + '\n')
                if build_dev and idx % (1 // dev_ratio) == 0:
                    dev_file.write(string + '\t' + DEFAULT + '\n')
        to_file.write('\n')
        if build_dev and idx % (1 // dev_ratio) == 0:
            dev_file.write('\n')
    to_file.close()
    from_file.close()
    print("clean -done.")
    if build_dev:
        dev_file.close()
        return to_path, dev_path
    else:
        return to_path, None


def write_clear_data_pd(from_path,
                        DEFAULT="O",
                        build_dev=False,
                        dev_ratio=0.4,
                        domain="communication",
                        BIO=False):
    """
    Given adequate stv data file, clean out the data that we need for sloting tagging
    Args:
        from_path: (string) path of original stv file
        build_dev: (bool) do you need to build dev from data file
        dev_ratio: (float) ratio of data build from data file
    Returns:
        (sting) path of clear data file
    """
    df = pd.read_csv(from_path, sep='\t')
    df_get = df[df.Domain == domain][["SlotString"]]
    print(df_get.dtype)
    lines = zip(df_get["SlotString"])

    to_path = os.path.splitext(from_path)[0] + '.txt'
    to_file = open(to_path, 'w', encoding='utf-8-sig')
    data_name = os.path.split(to_path)[1]
    print('begin clean dataset to {}...'.format(data_name))

    if build_dev:
        dev_path = os.path.join(
            os.path.split(to_path)[0], 'dev' + os.path.splitext(to_path)[1])
        dev_file = open(dev_path, 'w', encoding='utf-8-sig')
        print('meanwhile generate clear dev data from {}...'.format(data_name))
    for idx, line in enumerate(lines):
        label_chunk, S = False, []
        for string in line[0].split():
            m = re.match(r'^(</|<)([^<>]+)>$', string)
            if m is not None:
                label_chunk = not label_chunk
                label = m.group(2).lower()
                if not label_chunk:
                    if len(S) > 1:
                        for i in range(len(S)):
                            if i == 0:
                                S[i] = S[i][0] + "\tB-" + S[i][1] if BIO else S[i][0] + "\t" + S[i][1]
                            else:
                                S[i] = S[i][0] + "\tI-" + S[i][1] if BIO else S[i][0] + "\t" + S[i][1]
                    elif len(S) == 1:
                        S[0] = S[0][0] + "\t" + S[0][1]
                    to_file.write(''.join(S))
                    if build_dev and idx % (1 // dev_ratio) == 0:
                        dev_file.write(''.join(S))
                    S = []
            elif label_chunk:
                S += (string, label + '\n'),
            else:
                to_file.write(string + '\t' + DEFAULT + '\n')
                if build_dev and idx % (1 // dev_ratio) == 0:
                    dev_file.write(string + '\t' + DEFAULT + '\n')
        to_file.write('\n')
        if build_dev and idx % (1 // dev_ratio) == 0:
            dev_file.write('\n')
    to_file.close()
    print("clean -done.")
    if build_dev:
        dev_file.close()
        return to_path, dev_path
    else:
        return to_path, None


def cacl_line(file_path):
    with open(file_path, encoding="utf-8-sig") as f:
        lines = f.readlines()
        n = 0
        for i, line in enumerate(lines):
            if len(line) == 1:
                n += 1
    return n, len(lines)


def selc_data(from_path):
    df = pd.read_csv(from_path, sep='\t')
    df_get = df[df.Domain == "communication"][["Query", "SlotString"]]
    df_get.to_csv(
        os.path.splitext(from_path)[0] + "-communication.tsv",
        sep="\t",
        encoding="utf-8-sig",
        index=False,
        header=False)
    return os.path.splitext(from_path)[0] + "-communication.tsv"


def clear_data_path(from_path):
    return os.path.splitext(from_path)[0] + ".txt"


def _eval_help(tag, lt, lp, major=1):
    lt = list(map(lambda x: ["Neg", tag][x == tag], lt))
    lp = list(map(lambda x: ["Neg", tag][x == tag], lp))
    cache = [(p == tag) for i, p in enumerate([lt, lp][major])]
    pre, loc = cache[0], []
    for i, flag in enumerate(cache):
        if i == 0:
            l = i
        elif pre != flag:
            r = i
            loc += (l, r),
            l = i
        pre = flag
    loc += (l, len(cache)),
    lt_res, lp_res = [], []
    for i, t in enumerate(loc):
        lt_res += tuple(lt[t[0]:t[1]]),
        lp_res += tuple(lp[t[0]:t[1]]),
    return lt_res, lp_res


def infer_eval(filepath):
    with open(filepath, 'r', encoding="utf-8-sig") as f:
        lines = f.readlines()
    tag, pred = [], []
    eval_tag = {
        "TOTAL": {
            'N': 0,
            "TP": 0,
            "FN": 0,
            "FP": 0,
            "precision": 0,
            "recall": 0,
            "F1": 0
        }
    }
    for i, line in enumerate(lines):
        line = line.strip().split()
        if len(line) > 0:
            tag += line[1],
            pred += line[-1],
        else:
            kind = set(pred)
            for k in kind:
                if k not in eval_tag:
                    eval_tag[k] = {
                        'N': 0,
                        "TP": 0,
                        "FN": 0,
                        "FP": 0,
                        "precision": 0,
                        "recall": 0,
                        "F1": 0
                    }
                tag_chunk, pred_chunk = _eval_help(k, tag, pred)
                #                 if k=="title":print(i,tag_chunk,pred_chunk)
                for i, p in enumerate(pred_chunk):
                    if k in p and tag_chunk[i] != pred_chunk[i]:
                        eval_tag[k]["FP"] += 1

            kind = set(tag)
            for k in kind:
                if k not in eval_tag:
                    eval_tag[k] = {
                        'N': 0,
                        "TP": 0,
                        "FN": 0,
                        "FP": 0,
                        "precision": 0,
                        "recall": 0,
                        "F1": 0
                    }
                tag_chunk, pred_chunk = _eval_help(k, tag, pred, 0)
                #                 if k=="title":print(tag_chunk,pred_chunk)
                for i, p in enumerate(tag_chunk):
                    if k in p:
                        if tag_chunk[i] != pred_chunk[i]:
                            eval_tag[k]["FN"] += 1
                        else:
                            eval_tag[k]["TP"] += 1

            tag, pred = [], []
    for k in eval_tag.keys():
        if k == "O":
            continue
        eval_tag["TOTAL"]["TP"] += eval_tag[k]["TP"]
        eval_tag["TOTAL"]["FP"] += eval_tag[k]["FP"]
        eval_tag["TOTAL"]["FN"] += eval_tag[k]["FN"]

    for k, v in eval_tag.items():
        v["N"] = v["TP"] + v["FN"]
        v["precision"] = round(v["TP"] / (v["TP"] + v["FP"]) * 100,
                               2) if v["TP"] + v["FP"] > 0 else "NaN"
        v["recall"] = round(v["TP"] / (v["TP"] + v["FN"]) * 100,
                            2) if v["TP"] + v["FN"] > 0 else "NaN"
        v["F1"] = round(2 * v["TP"] / (2 * v["TP"] + v["FP"] + v["FN"]) * 100,
                        2) if 2 * v["TP"] + v["FP"] + v["FN"] > 0 else "NaN"
    infer_file = os.path.splitext(filepath)[0]
    if infer_file[-3] == ".":
        infer_file = infer_file[:-19]
    file_name = infer_file + "summary.%s.txt" % datetime.datetime.now(
    ).isoformat()[:19].replace(":", ".")
    percent = set(["precision", "recall", "F1"])
    params = ["N", "TP", "FN", "FP", "precision", "recall", "F1"]
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        f.write("Result:\tslot ")
        S = []
        for k, v in eval_tag["TOTAL"].items():
            S += k + "=" + str(v) + "%" * (k in percent),
        f.write(", ".join(S) + "\n")
        d = sorted(eval_tag.items(), key=lambda kv: kv[1]["N"], reverse=True)
        f.write("===========Slot-Specific Metrics:=========\n")
        for tag_items in d:
            if tag_items[0] == "TOTAL" or tag_items[0] == "O":
                continue
            S = []
            for param in params:
                for k, v in tag_items[1].items():
                    if k == param:
                        S += k + "=" + str(v) + "%" * (param in percent),
            f.write(tag_items[0] + ":\t" + ", ".join(S) + "\n")
    return file_name


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)
    
    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [
                    v * (current - self.seen_so_far),
                    current - self.seen_so_far
                ]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (
                        k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (
                        k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)
