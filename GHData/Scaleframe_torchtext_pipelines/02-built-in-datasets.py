import torch
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

'''
To install spacy languages use:
python -m spacy download en
python -m spacy download de
'''

spacy_en = spacy.load('en') 
spacy_ger = spacy.load('de')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

# preprocess fields
english = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, lower=True)

# train test split
train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                                fields=(german, english))

# build vocab
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=64,
    device=device
)

for batch in train_iterator:
    print(batch)

