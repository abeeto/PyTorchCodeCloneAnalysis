import gc
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
import transformers
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers.modeling_bert import BertModel
from torch.utils.tensorboard import SummaryWriter

MAX_LEN = 200
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-05

tokenizer = transformers.BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
train_size = 0.8
valid_size = 0.5

max_length = 0
train_texts = []
train_labels = []

csv_counter = 0
with open('training_data.csv', 'r', encoding='utf-8') as csv_read:
    csv_reader = csv.reader(csv_read)
    for csv_read_row in csv_reader:
        csv_counter += 1
        if csv_counter > 1:
            train_texts.append(csv_read_row[1])
            train_labels.append(int(csv_read_row[0]))
            if len(csv_read_row[1]) > max_length:
                max_length = len(csv_read_row[1])

data = {'texts':train_texts, 'labels':train_labels}
new_df = pd.DataFrame(data)

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe.texts
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
       texts = str(self.texts[index])
        texts = " ".join(texts.split())

        inputs = self.tokenizer.encode_plus(
            texts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

sample_df=new_df.sample(frac=1,random_state=200)
train_dataset = sample_df.sample(frac=train_size,random_state=200)
test_df = sample_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

valid_dataset = test_df.sample(frac=valid_size,random_state=200)
test_dataset = test_df.drop(valid_dataset.index).reset_index(drop=True)
valid_dataset = valid_dataset.reset_index(drop=True)

print(f'Dataset: {sample_df.shape} Train: {train_dataset.shape} Validation: {valid_dataset.shape} Test: {test_dataset.shape}')

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
valid_params  = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **valid_params)
testing_loader = DataLoader(testing_set, **test_params)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(768, 384)
        self.l4 = torch.nn.BatchNorm1d(384)
        self.l5 = torch.nn.Linear(384, 96)
        self.l6 = torch.nn.Dropout(0.2)
        self.l7 = torch.nn.Linear(96, 24)
        self.l8 = torch.nn.Dropout(0.2)
        self.l9 = torch.nn.Linear(24, 2)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = F.relu(self.l5(output_4))
        output_6 = self.l6(output_5)
        output_7 = F.relu(self.l7(output_6))
        output_8 = self.l8(output_7)
        output = self.l9(output_8)
        return output

model = BERTClass()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.AdamW(params= model.parameters(), lr=LEARNING_RATE)
STEP_SIZE = int(TRAIN_SIZE / TRAIN_BATCH_SIZE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.75)

def images_to_probs(model, ids, mask, token_type_ids):
    output = model(ids, mask, token_type_ids)
    _, preds_tensor = torch.max(output.cpu(), 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(model, ids, mask, token_type_ids, labels):
    classes = ['Label 1', 'Label 2']
    
    preds, probs = images_to_probs(model, ids, mask, token_type_ids)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(2):
        ax = fig.add_subplot(1, 2, idx+1, xticks=[], yticks=[])
        labels_idx = np.argmax(labels[idx].cpu())
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[int(preds[idx])],
            probs[idx] * 100.0,
            classes[labels_idx]),
                    color=("green" if int(preds[idx])==labels_idx.item() else "red"))
    return fig

%load_ext tensorboard
writer = SummaryWriter('tensorboard')
gc.collect()
torch.cuda.empty_cache()

def train():
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        for _,data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            running_loss += loss.item()
            if _ == len(training_loader) - 1:
                average_loss = running_loss / ( len(training_loader) - 1)
                print(f'Epoch: {epoch + 1}, Loss: {average_loss}, lr: {scheduler.get_last_lr()[0]}')
                writer.add_scalar('training loss', running_loss / 1000, epoch * len(training_loader) + _)
                writer.add_figure('predictions vs. actuals', plot_classes_preds(model, ids, mask, token_type_ids, targets), global_step=epoch * len(training_loader) + _)
                running_loss = 0.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        fin_valid_targets=[]
        fin_valid_outputs=[]
        
        with torch.no_grad():
            for _,valid_data in enumerate(validation_loader, 0):
                valid_ids = valid_data['ids'].to(device, dtype = torch.long)
                valid_mask = valid_data['mask'].to(device, dtype = torch.long)
                valid_token_type_ids = valid_data['token_type_ids'].to(device, dtype = torch.long)
                valid_targets = valid_data['targets'].to(device, dtype = torch.float)

                valid_outputs = model(valid_ids, valid_mask, valid_token_type_ids)
                
                fin_valid_targets.extend(valid_targets.cpu().detach().numpy().tolist())
                fin_valid_outputs.extend(torch.sigmoid(valid_outputs).cpu().detach().numpy().tolist())
        
        arg_valid_outputs = np.array(fin_valid_outputs).argmax(axis=1)[:,None] == range(np.array(fin_valid_outputs).shape[1])
        
        valid_accuracy = metrics.accuracy_score(fin_valid_targets, arg_valid_outputs)
        print(f"Validation Accuracy = {valid_accuracy}")
    
    writer.add_graph(model, (ids, mask, token_type_ids))
    writer.close()

train()

def testing():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

test_outputs, test_targets = testing()

outputs = np.array(test_outputs).argmax(axis=1)[:,None] == range(np.array(test_outputs).shape[1])

accuracy = metrics.accuracy_score(test_targets, outputs)
f1_score_micro = metrics.f1_score(test_targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(test_targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
