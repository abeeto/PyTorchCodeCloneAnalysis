import json
import codecs
import subprocess
import re
import torchtext.data as data 

dataset_path = '/Users/alvin/NCC/git_repository/multi30k/data/task1/raw/raw_data/'
de_path_train = dataset_path + 'train.de'
de_path_val = dataset_path + 'val.de'
de_path_test = dataset_path + 'test_2016_flickr.de'
en_path_train = dataset_path + 'train.en'
en_path_val = dataset_path + 'val.en'
en_path_test = dataset_path + 'test_2016_flickr.en'

def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = f.read().splitlines()
    return content

def write_csv(filename, zip_deen):
    with open(filename, 'w', encoding='utf-8') as fp:
        output = '\n'.join('{}\t{}'.format(de,en) for de,en in zip_deen)
        fp.write(output)
    
train_de = read_text(de_path_train)
train_en = read_text(en_path_train)
valid_de = read_text(de_path_val)
valid_en = read_text(en_path_val)
test_de = read_text(de_path_test)
test_en = read_text(en_path_test)

#train = list(map(list,zip(train_de, train_en)))
#val   = list(map(list,zip(valid_de, valid_en)))
#test  = list(map(list,zip(test_de, test_en)))
write_csv('./multi30k_csv/train.csv', zip(train_de,train_en))
write_csv('./multi30k_csv/valid.csv', zip(valid_de,valid_en))
write_csv('./multi30k_csv/test.csv', zip(test_de,test_en))

'''
with open('./json_folder/train.csv', 'w', encoding='utf-8') as fp:
     output = '\n'.join('{}\t{}'.format(de,en) for de,en in zip(train_de, train_en))
     fp.write(output)

with open('./json_folder/valid.csv', 'w', encoding='utf-8') as fp:
     output = '\n'.join('{}\t{}'.format(de,en) for de,en in zip(valid_de, valid_en))
     fp.write(output)

with open('./json_folder/test.csv', 'w', encoding='utf-8') as fp:
     output = '\n'.join('{}\t{}'.format(de,en) for de,en in zip(test_de, test_en))
     fp.write(output)
'''
'''
with codecs.open('train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(train_list,f,indent=4,ensure_ascii=False))
'''
'''
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
#fields = {"DE":('de',DE),"EN":('en',EN)}
fields = [("de", DE), ("en", EN)] 

train_examples = list(map(lambda x:data.Example.fromlist(x, fields), train))
train_dataset = data.Dataset(examples=train_examples,fields=fields)
#train_dataset = data.Dataset(train_examples,fields)
DE.build_vocab(train_dataset)
EN.build_vocab(train_dataset)
train_bucketiter = data.BucketIterator(dataset=train_dataset,batch_size=32, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))
print(train_batch)
print('train_batch.en ==> col-basd data format \n' + str(train_batch.en))
# data is saved as col-based rather than row-based, we should use [:,0] to retrive first sentence in first batch 
print('trian_batch.en[:,0] ==> first column \n ' + str(train_batch.en[:,0]))
first_en = list(map(lambda x: EN.vocab.itos[x],train_batch.en[:,0]))
first_de = list(map(lambda x: DE.vocab.itos[x],train_batch.de[:,0]))
print(first_en)
print(first_de)
#verify value is corrected
print('train_examples[0].en ==> ' + str(train_examples[0].en))
print('train_examples[0].de ==> ' + str(train_examples[0].de))
'''