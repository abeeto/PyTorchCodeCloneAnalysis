import re
import spacy
import codecs
import subprocess
import jieba
from torchtext.data import Field, Example, BucketIterator
from torchtext.data import Dataset,TabularDataset
from torchtext.datasets import Multi30k


def load_dataset(batch_size, macbook=False):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]
    
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN

def load_dataset_txt(batch_size,macbook=False):
    DE = Field(include_lengths=True,
                init_token='<sos>', eos_token='<eos>')
    EN = Field(include_lengths=True,
                init_token='<sos>', eos_token='<eos>')
    fields = {"DE":('src',DE),"EN":('trg',EN)}
    train, val, test = TabularDataset.splits(path='./multi30k_json/',format='json',fields=fields,
                                                train='train.json',validation='valid.json',test='test.json')
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]

    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN

def get_xmllcsts(filename,limit=None):
    # regular_expression is suitable for LCSTS PART_I.txt, PART_II.txt, PART_III.txt
    pattern = re.compile(r'''<doc id=(?:\d+)>(?:\n\s+<human_label>(?:\d+)</human_label>)?
    <summary>\n\s+(.+)\n\s+</summary>
    <short_text>\n\s+(.+)\n\s+</short_text>\n</doc>''', re.M)
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = ''.join(f.readlines())
    lcsts_list = re.findall(pattern, content)[:limit]

    return lcsts_list

def jieba_tokenizer(text): # create a tokenizer function
    #return [tok.text for tok in spacy_en.tokenizer(text)]
    return [tok for tok in jieba.lcut(text)]

def load_dataset_lcsts(batch_size,macbook=False,filename=None):
    filename = './lcsts_xml/PART_I_10000.txt' if filename == None else filename
    TRG = Field(tokenize=jieba_tokenizer, include_lengths=True,
                    init_token='<sos>', eos_token='<eos>')
    SRC = Field(tokenize=jieba_tokenizer, include_lengths=True,
                    init_token='<sos>', eos_token='<eos>')
    fields = [('trg',TRG),('src',SRC)] 
    lcsts_list = get_xmllcsts(filename)

    examples = list(map(lambda x :Example.fromlist(x,fields),lcsts_list))
    all_data = Dataset(examples=examples,fields=fields) 
    train,val, test = all_data.split(split_ratio=[0.8,0.1,0.1]) 
    # reduce corpus capasity for macbook testing
    if macbook == True:
        train.examples = train.examples[0:int(len(train.examples)/10)]
        val.examples = train.examples[0:int(len(val.examples)/10)]
        test.examples = train.examples[0:int(len(test.examples)/10)]

    SRC.build_vocab(train.src, min_freq=2)
    TRG.build_vocab(train.trg, max_size=10000)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False, shuffle=False)
    return train_iter, val_iter, test_iter, SRC, TRG

if __name__ == "__main__":
    filename =  './lcsts_xml/PART_I_10000.txt'
    load_dataset_lcsts(batch_size=32,macbook=False,filename=filename)
