from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_metric
import numpy as np
import torch
import os 
import time

time_start = time.time()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 导入数据集
raw_datasets = load_dataset("imdb")
# 数据预处理
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# 训练数据和测试数据集
print("分割数据集")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

# 定义模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
# GPU运行
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# 训练器
training_args = TrainingArguments("test_trainer")
print("metric")
metric = load_metric("./metrics/accuracy/accuracy.py")
print("trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
print("开始训练")
trainer.train()
print("开始评估")
trainer.evaluate()
time_end = time.time()

time_c = time_end - time_start
print("time cost", time_c, 's')
