import pandas as pd
import matplotlib.pyplot as plt
train_data=pd.read_csv("C:/Users/s123c/Desktop/train_ dataset/nCoV_100k_train.labled.csv",header=0,encoding='utf-8')
test_data=pd.read_csv("C:/Users/s123c/Desktop/train_ dataset/nCoV_100k_train.labled.csv",header=0,encoding='utf-8')
print(train_data.head())
train_data['情感倾向'].value_counts().plot.bar()
plt.title('sentiment')
#plt.show()
import torch
import pandas as pd
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import torch.utils.data.dataloader as dataloader
from sklearn.metrics import accuracy_score,recall_score,f1_score
from pytorch_pretrained_bert import BertForSequenceClassification,BertModel
from pytorch_pretrained_bert import BertAdam
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.model_selection import StratifiedKFold
import time
import random

MAX_SEQUENCE_LENGTH = 140
batch_size = 16
epochs = 9
input_categories = '微博中文内容'
output_categories = '情感倾向'
torch.backends.cudnn.enabled = False

train_df1 = train_data.copy()
train_df1=train_df1[~train_df1['情感倾向'].isin(['9','-','·'])]
train_df1 = train_df1.fillna(10)
train_df1=train_df1[~train_df1['情感倾向'].isin(['-2','10','4',10])]
'''
import zipfile

f = zipfile.ZipFile("E:/data/chinese_roberta_wwm_ext_pytorch.zip",'r')
for file in f.namelist():
  f.extract(file,'E:/data/chinese_roberta_wwm_ext_pytorch/')
f.close
'''
import random


def k_fold_split(train, k):
    os.system("mkdir data")
    k_fold = []
    index = set(range(train.shape[0]))
    for i in range(k):
        # 防止所有数据不能整除k，最后将剩余的都放到最后一折
        if i == k - 1:
            k_fold.append(list(index))
        else:
            tmp = random.sample(list(index), int(1.0 / k * train.shape[0]))
            k_fold.append(tmp)
            index -= set(tmp)
    # 将原始训练集划分为k个包含训练集和验证集的训练集，同时每个训练集中，训练集：验证集=k-1:1
    for i in range(k):
        print("第{}折........".format(i + 1))
        tra = []
        dev = k_fold[i]
        for j in range(k):
            if i != j:
                tra += k_fold[j]
        train.iloc[tra].to_csv("data/train_{}".format(i), sep=",", index=False)
        train.iloc[dev].to_csv("data/val_{}".format(i), sep=",", index=False)
    print("done!")


k_fold_split(train_df1, 5)

train_0 = pd.read_csv("data/train_1")
val_0 = pd.read_csv("data/val_1")
print(train_0.shape,val_0.shape)

train_0 = train_0[:1024]
val_0 = val_0[:128]

def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    def return_id(str1, truncation_strategy, length):
        inputs = tokenizer.tokenize(str1)
        if len(inputs) > 138:
            inputs = inputs[:138]
        inputs = ["[CLS]"] + inputs + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(inputs)
        #         print(input_ids)
        input_masks = [1] * len(input_ids)
        #         print(input_masks)
        input_segments = [0] * len(input_ids)
        padding_length = length - len(input_ids)
        #         padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([0] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        # if len(input_ids) != 200:
        #   print(str1,len(input_ids))
        return [input_ids, input_masks, input_segments]

    input_ids, input_masks, input_segments = return_id(instance, 'longest_first', max_sequence_length)
    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for instance in tqdm(df[columns]):
        ids, masks, segments = \
            _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    # print(input_ids)

    return input_ids, input_masks, input_segments

def compute_output_arrays(df,columns):
    return np.asarray(df[columns].astype(int) + 1)

def data_loader(input_ids,input_masks,input_segments,label_ids):
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_masks, dtype=torch.long)
    all_segment_ids = torch.tensor(input_segments, dtype=torch.long)
    all_label = torch.tensor(label_ids, dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    return train_dataloader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.bert = BertModel.from_pretrained("chinese_roberta_wwm_ext_pytorch/")
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='E:/data/chinese_roberta_wwm_ext_pytorch/')

        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, 3)

    def forward(self, input_ids, input_mask, segment_ids):
        _, pooled = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                              output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


tokenizer = BertTokenizer.from_pretrained('E:/data/chinese_roberta_wwm_ext_pytorch/vocab.txt')
t_input_ids, t_input_masks, t_input_segments = compute_input_arrays(train_0,input_categories,tokenizer,MAX_SEQUENCE_LENGTH)
print(len(t_input_segments))
t_label_ids = compute_output_arrays(train_0, output_categories)
train_dataloader = data_loader(t_input_ids, t_input_masks, t_input_segments,t_label_ids)


v_input_ids, v_input_masks, v_input_segments = compute_input_arrays(val_0,input_categories,tokenizer,MAX_SEQUENCE_LENGTH)
v_label_ids = compute_output_arrays(val_0, output_categories)
val_dataloader = data_loader(v_input_ids, v_input_masks, v_input_segments,v_label_ids)


device = torch.device("cuda:0")
# bert_model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='bert-base-chinese', num_labels=3)
bert_model = Model().to(device)
param_optimizer = list(bert_model.named_parameters())  # 模型参数名字列表
print(param_optimizer)
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
NUM_EPOCHS =8
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=0.05,
                     t_total=len(train_0) * NUM_EPOCHS)
# optimizer = BertAdam(bert_model.parameters(), lr=1e-6)
criterion = nn.CrossEntropyLoss()

def change_csv(abblist):
    abclist=[]
    for i in range(157):
      if i != 156 :
        for j in range(64):
            abclist.append(int(abblist[i][j])-1)
      else:
        for j in range(16):
          abclist.append(int(abblist[i][j])-1)
    dic1={}
    for i in abclist:
        dic1[i] = abclist.count(i)
    print(dic1)
    return abclist


test_df = pd.read_csv('C:/Users/s123c/Desktop/train_ dataset/nCoV_100k_train.labled.csv', header=0)
# test_df = pd.read_csv('nCov_10k_test.csv',header=0)
test_df1 = test_df.copy()
# test_df1.info()
dev_input_ids, dev_input_masks, dev_input_segments = compute_input_arrays(test_df1, input_categories, tokenizer,
                                                                          MAX_SEQUENCE_LENGTH)


def test_loader(input_ids, input_masks, input_segments):
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_masks, dtype=torch.long)
    all_segment_ids = torch.tensor(input_segments, dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


dev_dataloader = test_loader(dev_input_ids, dev_input_masks, dev_input_segments)
def create_test_csv(abclist):
    df1_test = test_df1.copy()
    df1_test['id']=df1_test["微博id"]
    df2_test_pud=df1_test.drop(labels=['微博id','微博发布时间',"发布人账号",'微博中文内容','微博图片','微博视频'],axis=1)
    letters_test_pud = ['id']
    df7_test=df2_test_pud[letters_test_pud]
    df7_test['y']=None
    df7_test['y'] = abclist
    # df_sub['id'] = df_sub['id'].apply(lambda x: str(x)+' ')
    # df7_test.to_csv('test_03341.csv',index=False, encoding='utf-8')

    return df7_test


from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score
class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        # print("X:",X)
        X_p = np.copy(X)
        # print("X_p:",type(X_p),"coef:",coef)
        X_p = X_p*coef
        ll = f1_score(y, np.argmax(X_p, axis=-1), average='macro')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1. for _ in range(3)]
        # print("initial_coef :",initial_coef)
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
        print("X_p:",X_p,"f1:",f1_score(y, np.argmax(X_p, axis=-1), average='macro'))
        return f1_score(y, np.argmax(X_p, axis=-1), average='macro')

    def coefficients(self):
        return self.coef_['x']

    def prt_coef(self):
        am= self.coef_
        print(am)
op = OptimizedF1()




def train(model, iterator, optimizer, criterion, device):
    start = time.time()
    model.train()
    epoch_loss = 0
    i = 0

    for input_ids, segment_ids, input_mask, label_ids in iterator:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        logits = model(input_ids, segment_ids, input_mask)
        logits2 = logits.cpu().detach()
        y_pred_notrick = logits.argmax(dim=1).cpu()
        # print("logits1:",logits.argmax(dim=1))
        model.zero_grad()
        # if i %600 == 0:
        #     print("---未测试时coef的值---------------")
        #     op.prt_coef()
        op.fit(logits2, label_ids.cpu())
        tips = Variable(torch.Tensor(op.coefficients()), requires_grad=True)
        # if i %600 == 0:
        #     print("---测试后coef的值---------------")
        #     op.prt_coef()
        logits = tips * (logits.cpu())
        y_pred_c = logits.argmax(dim=1).cpu()
        logits = logits.to(device)
        # print("logits2:",logits.argmax(dim=1))
        # y_pred_label = y_pred.cpu()
        loss = F.cross_entropy(logits, label_ids)
        epoch_loss += loss.cpu()
        # y_pred_c = y_pred.argmax(dim=1).cpu()
        # print(y_pred_c)
        label_ids_c = label_ids.cpu()
        # print(label_ids_c)
        if i % 300 == 0:
            # print("pred_notrick:",y_pred_notrick)
            # print("pred_trick:",y_pred_c)
            # print("label:",label_ids_c)
            # op.prt_coef()
            print("i", i, "loss", loss.cpu(), "train acc:", accuracy_score(y_pred_c, label_ids_c), "train rec:",
                  recall_score(y_pred_c, label_ids_c, average='macro'), "train f1",
                  f1_score(y_pred_c, label_ids_c, average='macro'))
        loss.backward()
        optimizer.step()
        i += 1
    end = time.time()
    runtime = end - start
    print('time: %.2f', runtime)
    return epoch_loss / len(iterator)


def deval(model, iterator, criterion, device):
    model.eval()
    abblist = []
    n = 0
    f1 = 0
    acc = 0
    rec = 0
    with torch.no_grad():
        # print("----------pred time the coef:----------")
        # op.prt_coef()
        for input_ids, segment_ids, input_mask, label_ids in iterator:
            n += 1
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            output = model(input_ids, segment_ids, input_mask)

            # logits2 = output.cpu().detach()

            # tips2 = Variable(torch.Tensor(op.coefficients()),requires_grad=True)
            logits = output.cpu()
            # print("-------------test-------------------")
            # print("label:",output.argmax(dim=1).cpu())
            # print("pred:",logits.argmax(dim=1).cpu())
            # print("-------------test-------------------")
            # op.prt_coef()
            y_pred_label = logits.argmax(dim=1).cpu()
            acc += accuracy_score(y_pred_label, label_ids)
            rec += recall_score(y_pred_label, label_ids, average='macro')
            f1 += f1_score(y_pred_label, label_ids, average='macro')
        print("train acc :", acc / n, "rec:", rec / n, "f1:", f1 / n, 'n:', n)


def pred(model, iterator, criterion, device):
    model.eval()
    abblist = []
    abclist = []
    with torch.no_grad():
        for input_ids, segment_ids, input_mask in iterator:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            output = model(input_ids, segment_ids, input_mask)
            # tips2 = Variable(torch.Tensor(op.coefficients()),requires_grad=True)
            # logits = tips2*(output.cpu())
            y_pred_label = output.argmax(dim=1).cpu()
            abblist.append(y_pred_label)
        abclist = change_csv(abblist)
        test_csv = create_test_csv(abclist)
    return test_csv

for i in range(epochs):
    train_loss = train(bert_model, train_dataloader, optimizer, criterion, device)
    lss = "roberta_model/p100_roberta_trick_"+str(i)+"_.pk1"
    if i == 0 :
      stat = train_loss
    if i != 0 :
      if stat-train_loss < 0.005 :
        break
      if stat-train_loss < 0 :
        break
    torch.save(bert_model.state_dict(), lss)
    print("train loss: ", train_loss)
    deval(bert_model, val_dataloader, criterion, device)
    test_csv = pred(bert_model, dev_dataloader, criterion, device)
    test_csv.to_csv('roberta_model/test_bwetick_'+str(i)+'.csv',index=False, encoding='utf-8')
    # deval(bert_model, train_dataloader, criterion, device)
torch.save(bert_model.state_dict(), "roberta_model/p100_roberta_trick_true_end.pk1")

