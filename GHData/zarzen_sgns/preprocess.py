from nltk.tokenize import word_tokenize
import collections
import pickle

unk_sym = '<UNK>'


def load_corpus(filename):
  corpus_txt = []
  with open(filename, 'r') as input_file:
    for line in input_file:
      words = word_tokenize(line)
      corpus_txt.extend(words)

  return corpus_txt


def build_dictionary(corpus_txt, max_voc):
  dictionary = {unk_sym:0}

  word_counter = collections.Counter(corpus_txt)
  most_common_words = word_counter.most_common(max_voc - 1)
  most_common_words = sorted(most_common_words, key=lambda word:word[0])

  for w, _ in most_common_words:
    dictionary[w] = len(dictionary)

  reversed_dictionary = dict(zip(dictionary.values(),
                                 dictionary.keys()))

  print('length of dictionary', len(dictionary))
  return dictionary, reversed_dictionary


def txt2idx(corpus_txt, dictionary):
  corpus_idx = []
  for w in corpus_txt:
    if w in dictionary:
      corpus_idx.append(dictionary[w])
    else:
      corpus_idx.append(dictionary[unk_sym])
  return corpus_idx

def build_training_pairs(corpus_idx, window_size):
  w_idx = 0
  train = []
  for target_w in corpus_idx:
    context_words = corpus_idx[w_idx + 1: w_idx + 1 + window_size]
    context_words += corpus_idx[w_idx - window_size: w_idx]
    w_idx += 1
    for ctx_w in context_words:
      if target_w == 0 or ctx_w == 0:
        pass
      else:
        train.append((target_w, ctx_w))

  return train

def build_dataset(corpus_file):
  corpus_txt = load_corpus(corpus_file)
  word2idx, idx2word = build_dictionary(corpus_txt, 20000)
  corpus_idx = txt2idx(corpus_txt, word2idx)

  train_pairs = build_training_pairs(corpus_idx, 1)
  pickle.dump(train_pairs, open('train_pairs.p', 'wb'))
  pickle.dump((word2idx, idx2word), open('dictionary.p', 'wb'))

if __name__ == '__main__':
  build_dataset('corpus.txt')
