# Steps: 

# 1. Specify how preprocessing should be done -> Fields
# 2. Use Dataset to load the data -> TabularDataset
# 3. Contstruct an iterator to do batching and padding -> BucketIterator

import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy

spacy_en = spacy.load('en')

# 1. Specify how preprocessing should be done:
def tokenize(text): 
    return [tok.text for tok in spacy_en.tokenizer(text)]

quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {'quote': ('q', quote), 'score':('s', score)}

# 2. Use Dataset to lead the data:
# (See example data files in mydata directory)

# json data format
train_data, test_data = TabularDataset.splits(
                                        path='mydata',
                                        train='train.json',
                                        test='test.json',
                                        # validation='validation.json',
                                        format='json',
                                        fields=fields)

# # csv data format
# train_data, test_data = TabularDataset.splits(
#                                         path='mydata',
#                                         train='train.csv',
#                                         test='test.csv',
#                                         validation='validation.csv'
#                                         format='csv',
#                                         fields=fields)

# # tsv data format
# train_data, test_data = TabularDataset.splits(
#                                         path='mydata',
#                                         train='train.tsv',
#                                         test='test.tsv',
#                                         validation='validation.tsv'
#                                         format='tsv',
#                                         fields=fields)


# Build vocab from corpus. "Quotes", from example files
quote.build_vocab(train_data, 
                  max_size=10000,
                #   vectors='glove.68.100d',  
                  min_freq=1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Construct an iterator to do batching a padding:
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=2, 
    device=device)


# train_iterator
for batch in train_iterator:
    print(batch.q)
    # output is 2 tensors, of shape(14,2), (14,1) (batch size 2) 
    # represents numericalized quote field
    print(batch.s)
    # shape (1, 2), (1,1) (batch size 2) 
    # numericalized score field. 

