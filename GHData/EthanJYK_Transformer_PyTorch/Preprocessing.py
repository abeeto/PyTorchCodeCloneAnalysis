# Load Raw Text --------------------------------------------------------------#
import random

en_raw = open('./english.txt', encoding='utf-8').read().split('\n')
fr_raw = open('./french.txt', encoding='utf-8').read().split('\n')

# Get smaller samples for test purpose
random.seed(1)
sample_index = random.sample(range(len(en_raw)), 50000)
en_sample = [en_raw[i] for i in sample_index]
fr_sample = [fr_raw[i] for i in sample_index]
#-----------------------------------------------------------------------------#


# Tokenizer ------------------------------------------------------------------#
import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset

en = spacy.load('en')
fr = spacy.load('fr')

def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

def tokenize_fr(sentence):
    return [tok.text for tok in fr.tokenizer(sentence)]

# Field defines a datatype together with its instructions for converting to Tensor
EN_TEXT = Field(tokenize=tokenize_en)
FR_TEXT = Field(tokenize=tokenize_fr, init_token="<sos>", eos_token="<eos>")
#-----------------------------------------------------------------------------#


# Prepare Data ---------------------------------------------------------------#
import pandas as pd

# create a data frame
raw_data = {'English' : [line for line in en_sample], 'French': [line for line in fr_sample]}
df = pd.DataFrame(raw_data, columns=["English", "French"])

# remove very long sentences and sentences where translations are not of roughly equal length
df['eng_len'] = df['English'].str.count(' ') # count spaces
df['fr_len'] = df['French'].str.count(' ')
df = df.query('fr_len < 80 & eng_len < 80') 
df = df.query('fr_len < eng_len * 1.5 & fr_len * 1.5 > eng_len')

# split train and validation_set# split train, validation and test set
from sklearn.model_selection import train_test_split
train, val = train_test_split(df, test_size = 0.2)
val, test = train_test_split(val, test_size = 0.5)
train.to_csv("train.csv", index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)

# Generator
# associate the text in the 'English' column with the EN_TEXT field, and 'French' with FR_TEXT
train, val, test = TabularDataset.splits(path='./', train='train.csv',
                                   validation='val.csv', test='test.csv', format='csv',
                                   fields=[('English', EN_TEXT), ('French', FR_TEXT)])

# index all tokens
EN_TEXT.build_vocab(train, val, test)
FR_TEXT.build_vocab(train, val, test)

# see examples
print(EN_TEXT.vocab.stoi['the'])
print(EN_TEXT.vocab.itos[6])

# Simple Bucket Iterator - the lambda function tells the iterator to try and find sentences of the same length
train_iter = BucketIterator(train, batch_size=64, sort_key=lambda x: len(x.French), shuffle=True)
val_iter = BucketIterator(val, batch_size=64, sort_key=lambda x: len(x.French), shuffle=True)
test_iter = BucketIterator(test, batch_size=64, sort_key=lambda x: len(x.French), shuffle=True)

# see examples
batch = next(iter(train_iter))
print(batch.English)
#-----------------------------------------------------------------------------#



# Data Iterator --------------------------------------------------------------#
# Custom Batching Code - changing batch_size depending on the sequence length - will replace the previous one
# code from http://nlp.seas.harvard.edu/2018/04/03/attention.html
from torchtext import data

global max_src_in_batch, max_trg_in_batch

def batch_size_fn(new, count, sofar):
    '''
    Function of three arguments (new example to add, current count of examples 
    in the batch, and current effective batch size) that returns the new effective batch size 
    resulting from adding that example to a batch
    '''
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_trg_in_batch # define global vars
    if count == 1:
        max_src_in_batch = 0 # initialize
        max_trg_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.English)) # get longer one
    max_trg_in_batch = max(max_trg_in_batch, len(new.French) + 2) # + 2 for <sos>, <eos>
    src_elements = count * max_src_in_batch
    trg_elements = count * max_trg_in_batch
    return max(src_elements, trg_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100): # data.batch(data, batch_size, batch_size_fn=None)
                    p_batch = data.batch(
                            sorted(p, key=self.sort_key),
                            self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


train_iter = MyIterator(train, batch_size=1300, device='cuda:0', repeat=False,
                        sort_key= lambda x: (len(x.English), len(x.French)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)

val_iter = MyIterator(val, batch_size=1300, device='cuda:0', repeat=False,
                        sort_key= lambda x: (len(x.English), len(x.French)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)

test_iter = MyIterator(test, batch_size=1300, device='cuda:0', repeat=False,
                        sort_key= lambda x: (len(x.English), len(x.French)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
#-----------------------------------------------------------------------------#
