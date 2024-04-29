import os
import time
import argparse
import numpy as np
import torch
from sklearn import model_selection
from sklearn.preprocessing import normalize
from utils import print, ProgressBar, clean_str
from models.GloVe import GloVe, run_GloVe
from model import SentenceClassifier, run_SentenceClassifier

##########################################################################
#                         Setup                                          #
##########################################################################

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='CNN-Sentence-Classification-PyTorch')

parser.add_argument('--word-edim', type=int, default=300, help='word embedding dimension (default: 300)')

parser.add_argument('--glove-batch-size', type=int, default=4096, help='input batch size for training glove model (default: 4096)')
parser.add_argument('--glove-epochs', type=int, default=10, help='number of epochs for training glove model (default: 10)')
parser.add_argument('--glove-context-size', type=int, default=5, help='context size for training glove model (default: 5)')
parser.add_argument('--glove-x-max', type=int, default=100, help='max of x parameter for training glove model (default: 100)')
parser.add_argument('--glove-process-num', type=int, default=1, help='number of the preprocess corpus in training glove model (default: 1)')
parser.add_argument('--glove-alpha', type=float, default=0.75, help='alpha parameter for training glove model (default: 0.75)')
parser.add_argument('--glove-lr', type=float, default=0.1, help='learning rate for training glove model (default: 0.1)')

parser.add_argument('--cnn-output-channel', type=int, default=100, help='(default: 100)')
parser.add_argument('--cnn-n-gram-list', type=int, nargs='+', default=[3,4,5], help='(default: 100)')

parser.add_argument('--classifier-class-number', type=int, default=2, help='(default: 2)')
parser.add_argument('--classifier-dropout-rate', type=float, default=0.5, help='(default: 0.5)')
parser.add_argument('--classifier-lr', type=float, default=0.01, help='(default: 0.01)')
parser.add_argument('--classifier-grad-norm-clip', type=float, default=3, help='(default: 3)')
parser.add_argument('--classifier-batch-size', type=int, default=50, help='(default: 50)')
parser.add_argument('--classifier-epochs', type=int, default=50, help='(default: 50)')

parser.add_argument('--skip-glove', type=bool, default=False, help='(default: false)')
parser.add_argument('--embed-rand', type=bool, default=False, help='(default: false)')
parser.add_argument('--embed-normalize', type=bool, default=False, help='(default: false)')
parser.add_argument('--embed-pad-zero', type=bool, default=False, help='(default: false)')
parser.add_argument('--embed-static', type=bool, default=False, help='(default: false)')

parser.add_argument('--show-progress', type=bool, default=True, help='show progress (default: true)')
parser.add_argument('--no-cuda', type=bool, default=False, help='disables CUDA training (default: false)')
parser.add_argument('--random-state', type=int, default=0, help='random state (default: 0)')
parser.add_argument('--gpu-num', type=int, default=1, help='gpu number (default: 1)')
parser.add_argument('--data-dir', type=str, default='./data/rt-polaritydata/', help='dataset directory (default: ./data/rt-polaritydata/)')

config = parser.parse_args()

config.cuda = not config.no_cuda and config.gpu_num and torch.cuda.is_available()
if config.cuda: 
    print('Use {} GPU {}'.format(config.gpu_num, os.environ["CUDA_VISIBLE_DEVICES"]))
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) == config.gpu_num

if not config.random_state: config.random_state = int(time.time())
np.random.seed(config.random_state)
torch.manual_seed(config.random_state)
if config.cuda: torch.cuda.manual_seed(config.random_state)

if not os.path.exists("./checkpoints/"): os.makedirs("./checkpoints/")

##########################################################################
#                         Methods                                        #
##########################################################################

def process_corpus(config):
    assert os.path.exists(config.data_dir+'rt-polarity.pos')
    assert os.path.exists(config.data_dir+'rt-polarity.neg')
    corpus_path = [config.data_dir+'rt-polarity.pos', config.data_dir+'rt-polarity.neg']
    encoding = 'ISO-8859-1'
    tokenized_corpus = list()
    dictionary = dict()
    max_length = 0
    for path in corpus_path:
        with open(path, encoding=encoding) as f:
            for line in f:
                line = clean_str(line.lower().strip())
                if len(line) > max_length: 
                    max_length = len(line)
                for word in line:
                    tokenized_corpus.append(word)
    config.seq_len = max_length # これがsequenceの固定長さとなる
    unique_word_list = np.concatenate((['<pad>'],np.unique(tokenized_corpus))) # <pad>をこれから入れるし，これが0番目となる．
    config.unique_word_size = len(unique_word_list)
    dictionary['word2idx'] = {word: index for index, word in enumerate(unique_word_list)}
    dictionary['idx2word'] = {index: word for index, word in enumerate(unique_word_list)}
    all_data = [[],[]]
    for i, path in enumerate(corpus_path):
        with open(path, encoding=encoding) as f:
            for line in f:
                new_line = list(map(dictionary['word2idx'].get, clean_str(line.lower().strip())))
                new_line += [dictionary['word2idx']['<pad>']]*(max_length-len(new_line))
                all_data[i].append(new_line)
    train_pos_data, test_pos_data = model_selection.train_test_split(all_data[0], test_size=.1, random_state=config.random_state)
    train_neg_data, test_neg_data = model_selection.train_test_split(all_data[1], test_size=.1, random_state=config.random_state)
    train_pos_data = np.array(train_pos_data)
    test_pos_data = np.array(test_pos_data)
    train_neg_data = np.array(train_neg_data)
    test_neg_data = np.array(test_neg_data)
    train_data = np.concatenate((train_pos_data,train_neg_data), axis=0)
    test_data = np.concatenate((test_pos_data,test_neg_data), axis=0)
    train_label = np.concatenate((np.ones(train_pos_data.shape[0], dtype=np.int),np.zeros(train_neg_data.shape[0], dtype=np.int)), axis=0)
    test_label = np.concatenate((np.ones(test_pos_data.shape[0], dtype=np.int),np.zeros(test_neg_data.shape[0], dtype=np.int)), axis=0)

    print("Train data num : {}, Test data num : {}.".format(train_label.shape[0],test_label.shape[0]))

    return tokenized_corpus, dictionary, train_data, train_label, test_data, test_label

##########################################################################
#                         Run                                            #
##########################################################################

if __name__ == '__main__':
    tokenized_corpus, dictionary, train_data, train_label, test_data, test_label = process_corpus(config)

    print("Config")
    print(str(config))

    if not config.skip_glove:
        glove = GloVe(config, tokenized_corpus, dictionary)
        if config.cuda:
            if config.gpu_num > 1:
                glove = torch.nn.DataParallel(glove, device_ids=list(range(config.gpu_num))).cuda()
                run_GloVe(config, glove)
                word_embedding_array = glove.module.embedding()
            else:
                glove = glove.cuda()
                run_GloVe(config, glove)
                word_embedding_array = glove.embedding()
        else:
            run_GloVe(config, glove)
            word_embedding_array = glove.embedding()
        raise Exception("Glove Training done")
    else:
        loaded_data = np.load('./checkpoints/word_embedding_{}.npz'.format(config.word_edim))
        word_embedding_array = loaded_data['word_embedding_array']
        assert dictionary == loaded_data['dictionary']

    # normalizeを入れた，零ベクトルにする処理はこの後にしないと明らかにエラー
    # <pad>のembed結果のための零ベクトルを入れる．
    # Gloveで<pad>は学習しないようにしてる．
    # Unique wordとしてembedding arrayの中に場所は確保しているけど，tokenized corpusの中にないので学習は進まない
    # この<pad>は零ベクトルとする．
    if config.embed_normalize:
        word_embedding_array = normalize(word_embedding_array, axis=1, norm='l1')
    if config.embed_pad_zero:
        word_embedding_array[dictionary["word2idx"]['<pad>']] = 0

    model = SentenceClassifier(config, word_embedding_array, dictionary)
    if config.cuda:
        if config.gpu_num > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(config.gpu_num))).cuda()
            run_SentenceClassifier(config, model, train_data, train_label, test_data, test_label)
        else:
            model = model.cuda()
            run_SentenceClassifier(config, model, train_data, train_label, test_data, test_label)
    else:
        run_SentenceClassifier(config, model, train_data, train_label, test_data, test_label)
