import argparse
import sys
import os
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from dataset import SentenceDataset
from metric import Metric
from loss import FocalLoss
from alarm_bot import ExamAlarmBot
from train import train
from train import eval

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=130)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--warmup_epoch_count', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=34)
parser.add_argument('--bert_model_name', type=str, default='beomi/kcbert-base')
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--loss_func', type=str, default='bce')
args = parser.parse_args()


# load params
batch_size = args.batch_size
num_epochs = args.num_epochs
num_warmup_epochs = args.warmup_epoch_count
data_path = args.data_dir
result_path = args.result_dir
train_name = "{}_{}_{}_{}".format(args.loss_func, int(args.alpha * 100), int(args.gamma * 10), datetime.now().strftime("%m%d-%H%M"))
train_path = os.path.join(result_path, train_name)


# dataloader
train_dataset = SentenceDataset(os.path.join(data_path, 'train_set.pkl'), args.bert_model_name)
val_dataset = SentenceDataset(os.path.join(data_path, 'val_set.pkl'), args.bert_model_name)
test_dataset = SentenceDataset(os.path.join(data_path, 'test_set.pkl'), args.bert_model_name)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
}

if not os.path.exists(train_path):
    os.mkdir(train_path)

num_steps_per_epoch = len(dataloaders['train'])
num_train_steps = num_steps_per_epoch * num_epochs
num_warmup_steps = num_steps_per_epoch * num_warmup_epochs

# init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(args.bert_model_name, num_labels=args.num_classes)
model.to(device)

loss_func = {
    'focal': FocalLoss(alpha=args.alpha, gamma=args.gamma),
    'bce': nn.BCELoss()
}

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = loss_func[args.loss_func]
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
writer = SummaryWriter(log_dir=os.path.join(train_path, 'logs'))

print('batch_size {}, num_epochs {}, step_per_epoch {}'.format(batch_size, num_epochs, len(dataloaders['train'])))
eval_history = []

# train loop
for epoch in range(num_epochs):
    print('Train Epoch {} / {}'.format(epoch+1, num_epochs))
    train_params = {
        'epoch': epoch, 'dataloader': dataloaders['train'],
        'device': device, 'optimizer': optimizer,
        'model': model, 'criterion': criterion,
        'scheduler': scheduler, 'tb_writer': writer
    }
    train_result = train(**train_params)
    eval_history.append(train_result)

    # validation
    val_params = {
        'epoch': epoch, 'dataloader': dataloaders['val'],
        'device': device, 'model': model,
        'criterion': criterion, 'tb_writer': writer
    }
    val_result = eval(**val_params)
    eval_history.append(val_result)

# test
test_params = {
    'epoch': epoch, 'dataloader': dataloaders['test'],
    'device': device, 'model': model,
    'criterion': criterion, 'tb_writer': writer
}
test_result = eval(**test_params)
eval_history.append(test_result)

writer.flush()
writer.close()

# 하이퍼파라미터 저장
with open(os.path.join(train_path, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f)

# 실험결과 저장
with open(os.path.join(train_path, 'result.txt'), 'w') as f:
    f.write('\n'.join([str(hist) for hist in eval_history]))

# 모델 저장하기
PATH = "state_dict_model.pt"
torch.save(model.state_dict(), os.path.join(train_path, PATH))

bot = ExamAlarmBot()
bot.send_msg('torch {} train is done, result : {}'.format(train_name, str(eval_history[-1])))

# # 불러오기
# model = Net()
# model.load_state_dict(torch.load(PATH))
# model.eval()