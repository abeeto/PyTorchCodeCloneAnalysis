import gc
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from pytorch_transformers import BertConfig,BertTokenizer,WarmupLinearSchedule
from tqdm import tqdm,trange
from modules.bert import BertClassifier
from datasets.datasets import SequenceDataset

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = max_split_size_mb:32

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_FILE_PATH = './datasets/Sarcasm_Headlines_Dataset.json'
BATCH_SIZE = 2
WARMUP_STEPS = 3
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8

def seed_everything(seed = 10):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

'''seed'''
seed_everything()

'''BERT config'''
# 加载Bert默认配置 需要进行必要更改
config = BertConfig(hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072, num_labels=2)

'''构造 BERT model'''
# 创造自定义的BERTClassifier模型
model = BertClassifier(config)
model.to(DEVICE)

'''初始化 tokenizer'''
# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

'''初始化 训练数据集'''
# 加载训练数据集并拆分为训练集 和 验证集
train_dataset = SequenceDataset(TRAIN_FILE_PATH,tokenizer)

# 验证拆分
validation_split = 0.2 # 选取20% 作为验证集
dataset_size = len(train_dataset)
indices = list(range(dataset_size))     # indices 是指数
split = int(np.floor(validation_split * dataset_size))
shuffle_dataset = True  # 将训练数据打乱

if shuffle_dataset :
    np.random.shuffle(indices)

# 训练集大小 和 验证集大小
train_indices,val_indices = indices[split:],indices[:split]

# 进行子集采样
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(train_dataset,batch_size=1,sampler=train_sampler)
val_loader = DataLoader(train_dataset,batch_size=1,sampler=validation_sampler)

print ('Training Set Size {}, Validation Set Size {}'.format(len(train_indices), len(val_indices)))

'''反向传播'''
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# Adam优化
optimizer = torch.optim.Adam([
    {'params':model.bert.parameters(),'lr':1e-5},
    {'params':model.classifier.parameters(),'lr':3e-4}
])

# 调节 learning_rate
# 为了防止随机初始化的学习率在一开始就很高或者梯度爆炸之类的，又或者一直很低学不到东西
# 这里使用了WarmupLinearSchedule，作用是让学习率平稳，从物理角度增强学习
# 线性预热，然后线性衰减。
# 在“warmup_steps”训练步骤中线性增加学习率从 0 到 1。
# 在剩余的“t_total - warmup_steps”步骤中线性降低学习率从 1. 到 0
scheduler = WarmupLinearSchedule(optimizer,warmup_steps=WARMUP_STEPS,t_total=len(train_loader) // GRADIENT_ACCUMULATION_STEPS*NUM_EPOCHS)

model.zero_grad()
# 更新迭代每一次epoch
epoch_iterator = trange(int(NUM_EPOCHS),desc="Epoch") 
training_acc_list,validation_acc_list = [],[] # 训练集争取列表，验证集正确列表



for epoch in epoch_iterator:

    epoch_loss = 0.0
    train_correct_total = 0
    
    # Training 循环
    train_iterator = tqdm(train_loader,desc="Train Iteration")
    for step,batch in enumerate(train_iterator):
        model.train(True)
        input = {
            'input_ids' : batch[0].to(DEVICE),
            'token_type_ids' : batch[1].to(DEVICE),
            'attention_mask' : batch[2].to(DEVICE)
        }

        labels = batch[3].to(DEVICE)
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        logits = model(**input)

        loss = criterion(logits,labels) / GRADIENT_ACCUMULATION_STEPS
        ######这里不知道
        loss.backward() # 反向传播
        epoch_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scheduler.step()
            optimizer.step()
            model.zero_grad()

        _,predicted = torch.max(logits.data,1)
        correct_reviews_in_batch = (predicted == labels).sum().item()
        train_correct_total += correct_reviews_in_batch
    
    print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))
    
    # 验证集循环
    with torch.no_grad():
        val_correct_total = 0
        model.train(False)
        val_iterator = tqdm(val_loader,desc="Validation Iteration")
        for step,batch in enumerate(val_iterator):
            inputs = {
                'input_ids': batch[0].to(DEVICE),
                'token_type_ids': batch[1].to(DEVICE),
                'attention_mask': batch[2].to(DEVICE)
            }

            labels = batch[3].to(DEVICE)
            logits = model(**inputs)

            _, predicted = torch.max(logits.data, 1)
            correct_reviews_in_batch = (predicted == labels).sum().item()
            val_correct_total += correct_reviews_in_batch

        training_acc_list.append(train_correct_total * 100 / len(train_indices))
        validation_acc_list.append(val_correct_total * 100 / len(val_indices))
        print('Training Accuracy {:.4f} - Validation Accurracy {:.4f}'.format(
            train_correct_total * 100 / len(train_indices), val_correct_total * 100 / len(val_indices)))

    
