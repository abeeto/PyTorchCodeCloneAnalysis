import numpy as np
import pandas as pd
import random
from tensorboardX import SummaryWriter
from transformers import BertModel, BertTokenizer
import torch
'''
import csv

with open('E:/data/SER/IEMOCAP/processed/processed_label.txt') as f:
    full_label = f.readlines()
full_label = [x.strip() for x in full_label]

data = []
with open('E:/data/SER/IEMOCAP/processed/processed_label.txt') as f:
    data = f.readlines()
with open('E:/data/SER/IEMOCAP/processed/Four_label.csv', 'w') as f:
    for i, label in enumerate(full_label):
        if label != '-1':
            f.write( data[i] )
data = []
with open('E:/data/SER/IEMOCAP/processed/processed_tran.csv') as f:
    csv_reader = csv.reader(f)
    data = [ x[1] for x in csv_reader ]
    print(data[14]+ '\n'+data[15])

with open('E:/data/SER/IEMOCAP/processed/Four_trans.csv', 'w',newline='') as f:
    for i, label in enumerate(full_label):
        if label != '-1':
            #f.write( data[i] + '\n')
            writer=csv.writer(f)
            writer.writerow([data[i]])
'''
#data_loader

import numpy as np
import pandas as pd
import random
from tensorboardX import SummaryWriter
from transformers import BertModel, BertTokenizer
import torch
df=pd.read_csv('E:/data/SER/IEMOCAP/processed/Four_trans.csv')
docs=[]
for text,label in zip(df['text'],df['label']):
  if(label!=-1):
    docs.append({'text':text,'label':label})
#print(docs)

random.shuffle(docs)
random.shuffle(docs)
random.shuffle(docs)
total_length=len(docs)
train_length=int(.9*total_length)
#divide into train and test
train_list=docs[0:train_length]
test_list=docs[train_length:]
#print('num of items for train ',len(train_list))
#print('num of items for test ',len(test_list))

#initialisation
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        print('init of linear is done')
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.xavier_uniform_(m.bias)

#bert-classification
from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False,
)
#print(model)
params = list(model.named_parameters())
optimizer = AdamW(model.parameters(),lr=2e-5,eps=1e-8)

from transformers import get_linear_schedule_with_warmup

NUM_EPOCHS = 4
writer = SummaryWriter(log_dir='/content/')
total_steps = len(train_list) * NUM_EPOCHS
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)

#train model
import random
total_steps = 1
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device('cuda:0')
model = model.to(device)
for epoch in range(NUM_EPOCHS):
    print(epoch)
    model.train()
    random.shuffle(train_list)
    for every_trainlist in train_list:
        label1=every_trainlist['label']
        label1=torch.tensor([label1])
        text=every_trainlist['text']
        #print("text",text)
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        model.zero_grad()
        input_ids = input_ids.to(device)
        label1=label1.to(device)
        loss = model(input_ids,labels=label1).loss
        logits=model(input_ids,labels=label1).logits
        #print(model(input_ids,labels=label1))
        #print(loss)
        #print('loss.item()',loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        _, preds = torch.max(logits, 1)
    #print('preds',preds)
        accuracy = torch.sum(preds == label1)
    #print('accuracy',accuracy)
        if total_steps % 10 == 0:
          with torch.no_grad():
            _, preds = torch.max(logits, 1)
            accuracy = torch.sum(preds == label1)
            print("total step",total_steps,accuracy)
            writer.add_scalar('training loss', loss.item(), total_steps)
            writer.add_scalar('training accuracy', accuracy.item(), total_steps)
        total_steps+=1

#test model
y_actu=[]
y_pred=[]
device = torch.device('cuda:0')
model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for every_test_list in test_list:
    label1=every_test_list['label']
    label1=torch.tensor([label1])
    text=every_test_list['text']
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
      loss, output = model(input_ids,labels=label1)
      _, preds = torch.max(output, 1)
      y_actu.append(label1.numpy()[0])
      y_pred.append(preds.numpy()[0])



#Save the Models

torch.save(model, '/model_text_bert.pt')
model=torch.load('/model_text_bert.pt')