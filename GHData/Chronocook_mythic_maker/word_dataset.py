import numpy as np
import codecs 
import nltk
from unidecode import unidecode
import torch
from word_embedding import WordEmbedding
from torch.utils.data import DataLoader, Dataset


class WordDataset(Dataset):
    
    def __init__(self, text_fn, embedding, chunk_size=12):
        self.embedding = embedding
        self.chunk_size = chunk_size
        
        # read the data
        # input_file = 'data/keywell_corpus.txt'
        fp = codecs.open(text_fn, 'r', 'utf-8')
        words = nltk.word_tokenize(fp.read())
        words = map(unidecode, words)

        # filter/preprocess words
        words = [word.replace(',', '') for word in words]
        words = [word.lower() for word in words]
        # split on hyphens
        for word in words:
            if '-' in word:
                dash_words = word.split('-')
                words.remove(word)
                words.extend(dash_words)
        words = [word for word in words if word]

        self.words = words
        
    def __len__(self):
        return len(self.words)
    
    def dim(self):
        return self.embedding.dim
    
    def get_chunk(self):
        chunk_words = []
        got_good_words = False
        while not got_good_words:
            sta_ind = np.random.randint(0, len(self) - self.chunk_size - 1)
            end_ind = sta_ind + self.chunk_size
            chunk_words = self.words[sta_ind:end_ind]
            got_good_words = all([word in embedding for word in chunk_words])
        vec_chunk = np.stack([self.embedding[word] for word in chunk_words])
        return torch.from_numpy(vec_chunk)
    
    def get_chunks(self, n_chunks):
        return torch.stack([self.get_chunk() for _ in range(n_chunks)])


if __name__ == '__main__':
    embedding_fn = '/Users/bkeating/nltk_data/embeddings/glove/glove.6B.100d.txt'
    embedding = WordEmbedding(embedding_fn)
    dataset = WordDataset('data/keywell_corpus.txt', embedding)

    chunk = dataset.get_chunk()
    print(chunk.size())

    chunks = dataset.get_chunks(20)
    print(chunks.size())
