import torch.nn as nn
import codecs
import json
import numpy as np
import torch
from gensim.models.keyedvectors import KeyedVectors


class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
 
    """
    def __init__(self, embed_size_src, embed_size_tgt, vocab, pretrained_embed_tgt=True, 
                pretrained_embed_src=True, 
                ):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size_src = embed_size_src
        self.embed_size_tgt = embed_size_tgt

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']
        
        self.source = nn.Embedding(len(vocab.src),self.embed_size_src, padding_idx = src_pad_token_idx)
        self.target = nn.Embedding(len(vocab.tgt) ,self.embed_size_tgt, padding_idx = tgt_pad_token_idx)

        if pretrained_embed_tgt:
            emb_matrix_tgt = self.get_embedding_matrix(vocab_src='target',vocab=vocab, embed_size=self.embed_size_tgt, 
                                                    w2v_file= 'eng_vie_data/glove.6B.50d.txt'
                                                    )
            self.target.weight = nn.Parameter(torch.tensor(emb_matrix_tgt,dtype=torch.float32))

            self.target.weight.requires_grad= False #freeze word2vec embedding

        if pretrained_embed_src:
            emb_matrix_src = self.get_embedding_matrix(vocab_src='source',vocab=vocab, embed_size=self.embed_size_src,
                                                    w2v_file = 'eng_vie_data/wiki.vi.model.bin')

            self.source.weight = nn.Parameter(torch.tensor(emb_matrix_src,dtype=torch.float32))

            self.source.weight.requires_grad= False

    @staticmethod
    def open_vocab(file_path = 'eng_vn_vocab.json', src='source'):
        entry = json.load(codecs.open(file_path, 'r',encoding='cp720'))

        if src == 'source':
            word2id = entry['src_word2id']
        elif src == 'target':
            word2id = entry['tgt_word2id']

        return word2id


    def get_embedding_matrix(self,vocab_src, vocab, embed_size, w2v_file):
        #open gloves
        if w2v_file[len(w2v_file)-3:] == 'txt':
            with codecs.open(w2v_file,'r', encoding="cp720") as f:
                words = set()
                word_to_vec_map = {}
                for line in f:
                    line = line.strip().split()
                    curr_word = line[0]
                    words.add(curr_word)
                    word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        elif w2v_file[len(w2v_file)-3:] == 'bin': #'wiki.vi.model.bin'
            word_to_vec_map = KeyedVectors.load_word2vec_format(w2v_file,binary=True)

        if vocab_src == 'target':
            emb_matrix = np.zeros((len(vocab.tgt), self.embed_size_tgt))

        elif vocab_src == 'source':
            emb_matrix = np.zeros((len(vocab.src), self.embed_size_src))

        for word, index in self.open_vocab(src=vocab_src).items():
            if word in word_to_vec_map:
                emb_matrix[index, :] =word_to_vec_map[word]

        print('load word embedding for {0} successfully from {1}'.format(vocab_src,w2v_file))

        return emb_matrix
