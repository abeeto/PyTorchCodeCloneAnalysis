from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_metric
import numpy as np
import torch
import os 
import time
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

#计时开始
time_start = time.time()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 导入数据集
raw_datasets = load_dataset("imdb")
# 数据预处理
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# 训练数据和测试数据集
print("分割数据集")

# print(tokenized_datasets["train"].column_names)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# print(tokenized_datasets["train"].column_names)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#训练和测试dataloader
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# for batch in train_dataloader:
#     break
# batch = {k: v.shape for k, v in batch.items()}
# print(batch)

# 定义模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
# GPU运行
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
#优化器
optimizer = AdamW(model.parameters(), lr=5e-5)
#学习率
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
#开始训练
print("开始训练")
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
#开始评估
print("开始评估")
metric= load_metric("./metrics/accuracy/accuracy.py")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())


#计时结束
time_end = time.time()

time_c = time_end - time_start
print("time cost", time_c, 's')

'''
输出结果：
{'accuracy': 0.878}
time cost 184.7304391860962 s
注意事项：在运行的时候，需要把之前数据中的cache删去。
'''
