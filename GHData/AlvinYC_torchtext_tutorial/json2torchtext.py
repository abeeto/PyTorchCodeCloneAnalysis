'''
import torchtext.data as data

DE = data.Field()
EN = data.Field()
# 可以直接修改 key 的名稱
fields = {"DE":('de',DE),"EN":('en',EN)}

train = data.TabularDataset(path='./multi30k_json/train_small.json',format='json',fields=fields)
valid = data.TabularDataset(path='./multi30k_json/valid.json',format='json',fields=fields)
test  = data.TabularDataset(path='./multi30k_json/test.json',format='json',fields=fields)

DE.build_vocab(train)
EN.build_vocab(train)

train_iter = data.BucketIterator(dataset=train, batch_size=4,shuffle=False)

t1 = next(iter(train_iter))

train_iter = data.Iterator(dataset=train, batch_size=32, shuffle=True)

'''

'''
import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
train_examples = train.examples
#----------------------
#train_dataset = data.Dataset(examples=train_examples,fields=fields)
DE.build_vocab(train)
EN.build_vocab(train)
train_iter = data.Iterator(dataset=train,batch_size=5,shuffle=False,train=True)
print(train_iter.__dict__.keys())
# content from next(iter(train_iter)) is embed-format such as train_batch.en[:,0] = tensor([15,1467,1313,...])
train_batch0 = next(iter(train_iter))
# content from next(train_iter.batch) is text-format such as train_batch[0].en = ['A,'man','in]
train_batch1 = next(train_iter.batches)
## torchtext.data.Dataset
# step 1: 準備資料
# 將 預先準備好的 torchtext.data.Example 與 torchtext.data.field 轉換成 torchtext.data.Dataset 格式
# 本來應該自備 torchtext.data.Example, 這裏用偷吃步先用 torchtext.data.TaburaDataset 取出 Example 部份
DE = data.Field()
EN = data.Field()
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
train_examples = train.examples
# step 2: 進行轉換
train_dataset = data.Dataset(examples=train_examples,fields=fields)


## torchtext.data.tabularDataset
train_tabular = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)

## torchtext.data.BucketIterator
train_tabular_Bucket = data.BucketIterator(dataset=train_tabular,batch_size=5,shuffle=False)
'''


import torchtext.data as data
DE = data.Field(is_target=False)
EN = data.Field(is_target=True)
fields = {"DE":('de',DE),"EN":('en',EN)}
train = data.TabularDataset(path='./multi30k_json/train.json',format='json',fields=fields)
#----------------------
train_examples = train.examples
train_dataset = data.Dataset(examples=train_examples,fields=fields)
DE.build_vocab(train_dataset)
EN.build_vocab(train_dataset)
train_bucketiter = data.BucketIterator(dataset=train_dataset,batch_size=32, sort_key=False, train=True, shuffle=False)
# retrive frist batche
train_batch = next(iter(train_bucketiter))
print(train_batch)
print('train_batch.en ==> col-basd data format \n' + str(train_batch.en))
print(train_batch.en[:,0])
# data is saved as col-based rather than row-based, we should use [:,0] to retrive first sentence in first batch 
print('trian_batch.en[:,0] ==> first column \n ' + str(train_batch.en[:,0]))
# using vacaburaty to maping from index to string(label)
first_en = list(map(lambda x: EN.vocab.itos[x],train_batch.en[:,0]))
first_de = list(map(lambda x: DE.vocab.itos[x],train_batch.de[:,0]))
print(first_en)
print(first_de)
#verify value is corrected
print('train.examples[0].en ==> ' + str(train.examples[0].en))
print('train.examples[0].de ==> ' + str(train.examples[0].de))




