"""
Our Goal here is to take raw text data and get it into a format we can use in torchtext. 

As we saw before, torchtext wants to read in files that are in json, csv, or tsv formats. So let's get there first. 
"""
import torch
import spacy # for language tokenizers
import pandas as pd # we'll use this for PRE pre processing
from torchtext.data import Field, BucketIterator, TabularDataset # use this for pre processing. 
from sklearn.model_selection import train_test_split # use for train test splitting our dataframe.  

# Data files can be found at: https://nlp.stanford.edu/projects/nmt/
    # WMT'14 English-German data [Medium], 
        # train.en and train.de. 


# First we need to get our data read in from our text files. 

# Read in file line by line (first 100 lines) to prevent memory overflow:
with open('mydata/WMT_train.en') as myfile:
    english_text = [next(myfile) for line in range(10000)]
# returns list of strings, english sentences

with open('mydata/WMT_train.de') as myfile:
    german_text = [next(myfile) for x in range(10000)]
# returns: list of strings, german sentences.

'''
We can use pandas dataframe to get a dictionary of text into a useable format for training.
'''

# First we need to get our text data into a dictionary. 
raw_data = {"English": [line for line in english_text[1:100]],
            "German": [line for line in german_text[1:100]]}
# returns: python dictionary


# From dictionary we can make DataFrame object:
df = pd.DataFrame(raw_data, columns=['English', 'German'])

# With dataframe we can use sklearn's train/test/split:
train, test = train_test_split(df, test_size=0.1)


# Now we can write our train / test split to json:
train.to_json('train.json', orient='records', lines=True)
test.to_json('test.json', orient='records', lines=True)
# returns: json file with structure: 
#                {"English": english sentence, 
#                "German": german_sentence}


# Or csv format: 
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
# returns: json file with structure: 
#               {"English": english sentence, 
#                "German": german_sentence}

# load the language pipelines
spacy_en = spacy.load('en')
spacy_ger = spacy.load('de')

# use the tokenizers from each language pipeline
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

# preprocess fields
english = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_ger, lower=True)


fields = {"English": ("eng", english), "German": ("ger", german)}

# train test split
train_data, test_data = TabularDataset.splits(
    path="", train="train.json", test="test.json", format="json", fields=fields
)

# build vocab
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)


device = "cuda" if torch.cuda.is_available() else "cpu"

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=32,
    device=device
)

for batch in train_iterator:
    print(batch)


