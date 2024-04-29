import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}     # initializing word to index dictionary to encode words prior to training
        self.idx2word = {}     # initializing index to word dictionary to decode predictions post training
        self.idx = 0     # inital mapping index set to 0

    def add_word(self, word):

        # if word not present in dictionary, add it to the dictionary and map it to the index
        if not word in self.word2idx:
            self.word2idx[word] = self.idx   # word -> index
            self.idx2word[self.idx] = word     # index -> word
            self.idx += 1   # increment counter by 1

    # call function called if instance of Vocabulary class acts like a function.
    # if a word is given, return the index of the word if it exists
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    # return length of the word to index function
    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)  # coco object holding annotations (5 annotations per image), image ids, categories and images
    counter = Counter()  # counter object from collections library to hold the word and corresponding word frequency.

    # total number of captions in the dataset. An image has 5 captions, therefore total captions = 414113
    ids = coco.anns.keys()
    for i, id in enumerate(ids):  # looping over all the captions
        caption = str(coco.anns[id]['caption'])  # a single caption at a time
        tokens = nltk.tokenize.word_tokenize(
            caption.lower())  # tokenize the caption, i.e get the word and its frequency
        counter.update(tokens)  # update the counter object with the tokens in every loop.

        # print output having processed every 1000 tokens
        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()  # create a vocabulary object
    vocab.add_word('<pad>')   # adding some checkpoint words like start, end, pad, etc to our vocabulary
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # just for understanding effect of __call__ function.
    # here the vocab object acts like a function. This will invoke the __call__ method
    print('Index of <unk>:', vocab('<unk>'))

    # Add the words to the vocabulary by iterating over them one by one
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)   # call the vocabulary builder function
    vocab_path = args.vocab_path    # path to store the vocabulary file

    # dump the vocabulary object into a pickle file
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions_train2014.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
