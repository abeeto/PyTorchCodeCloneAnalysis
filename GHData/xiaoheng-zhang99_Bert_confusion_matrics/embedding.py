#用BERT输出词向量
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

inputs = tokenizer('我是一个好人', return_tensors='pt')
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print('last_hidden_states:' ,last_hidden_states.shape)
pooler_output = outputs.pooler_output
print('---pooler_output: ', pooler_output.shape)
