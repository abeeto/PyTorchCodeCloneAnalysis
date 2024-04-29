# %%
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", \
    "sst2", "stsb", "wnli"]

# %%
task = "sst2"
model_checkpoint = "distilbert-base-uncased"
batch_size = 64
num_epochs = 20

# %%
from datasets import load_dataset, load_metric
actual_task = "mnli" if task =="mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric("glue", actual_task)

# %%
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there \
        are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
# %%
show_random_elements(dataset["train"])
# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# %%
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
sentence1_key, sentence2_key = task_to_keys[task]
# %%
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key],
         truncation=True)
preprocess_function(dataset['train'][:5])
# %%
encoded_dataset = dataset.map(preprocess_function, batched=True)
# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments
from transformers import Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
    num_labels=num_labels)


# %%
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

#%%
args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    # save_strategy="epoch",
    # save_total_limit="5",
    load_best_model_at_end=True, # when set true, save_strategy is ignored
    metric_for_best_model=metric_name,
    # greater_is_better=True, # default to True, if lower the better, set to False
)
# %%
import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

# %%
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset[validation_key],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# # %%
# trainer.train()
# # %%
# trainer.evaluate()
# # %%
# trainer.save_model("test-glue/best-model")
# %%
import torch

model = AutoModelForSequenceClassification.from_pretrained("test-glue/best-model", 
    num_labels=num_labels)


# %%
# from transformers import DataCollatorWithPadding

# def explain(model, test_str):
#     data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
#     # test_str = "The new design is awful!"
#     ex_tokens = tokenizer([test_str])
#     str_list = tokenizer.convert_ids_to_tokens(ex_tokens['input_ids'][0])
#     # print(str_list)
#     ex_tokens = data_collator(ex_tokens)
#     result = model(output_hidden_states=True, **ex_tokens)
#     logits = result.logits
#     pred = torch.argmax(logits)
#     layer_ids = [-2, -3, -4, -5, -6, -7]
#     expln_ = []
#     for layer_id in layer_ids:
#     # layer_id = -2
#         if layer_id == -2:
#             result.hidden_states[layer_id].retain_grad()
#             model.zero_grad()
#             logits[0][pred.item()].backward(retain_graph=True)
#             hs_grad = result.hidden_states[layer_id].grad
#             expln = (hs_grad * result.hidden_states[layer_id]).sum(dim=-1)
#             cls_grad = hs_grad[:,0,:]
#             cls_hs = result.hidden_states[layer_id][:,0,:]
#             expln_.append(expln)
#         else:
#             result.hidden_states[layer_id].retain_grad()
#             model.zero_grad()
            
#             cls_hs.backward(cls_grad, retain_graph=True)
#             hs_grad = result.hidden_states[layer_id].grad
#             expln = (hs_grad * result.hidden_states[layer_id]).sum(dim=-1)
#             cls_grad = hs_grad[:,0,:]
#             cls_hs = result.hidden_states[layer_id][:,0,:]
#             expln_.append(expln)
    
#     expln = sum(expln_)
#     return pred, expln, str_list

# %%
from transformers import DataCollatorWithPadding

def explain(model, test_str):
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    # test_str = "The new design is awful!"
    ex_tokens = tokenizer([test_str])
    str_list = tokenizer.convert_ids_to_tokens(ex_tokens['input_ids'][0])
    # print(str_list)
    ex_tokens = data_collator(ex_tokens)
    result = model(output_hidden_states=True, **ex_tokens)
    logits = result.logits
    pred = torch.argmax(logits)
    layer_ids = [-2, -3, -4, -5, -6, -7]
    expln_ = []
    for layer_id in layer_ids:
    # layer_id = -2
        if layer_id == -2:
            result.hidden_states[layer_id].retain_grad()
            model.zero_grad()
            logits[0][pred.item()].backward(retain_graph=True)
            
            hs_grad = result.hidden_states[layer_id].grad
            expln = (hs_grad * result.hidden_states[layer_id]).sum(dim=-1)
            cls_grad = hs_grad[:,0,:]
            cls_hs = result.hidden_states[layer_id][:,0,:]
            sep_grad = hs_grad[:,-1,:]
            sep_hs = result.hidden_states[layer_id][:,-1,:]

            expln_.append(expln)
        else:
            result.hidden_states[layer_id].retain_grad()

            model.zero_grad()
            cls_hs.backward(cls_grad, retain_graph=True)
            hs_grad = result.hidden_states[layer_id].grad
            expln = (hs_grad * result.hidden_states[layer_id]).sum(dim=-1)

            cls_grad = hs_grad[:,0,:]
            cls_hs = result.hidden_states[layer_id][:,0,:]

            model.zero_grad()
            sep_hs.backward(sep_grad, retain_graph=True)
            hs_grad = result.hidden_states[layer_id].grad
            expln += (hs_grad * result.hidden_states[layer_id]).sum(dim=-1)

            sep_grad = hs_grad[:,-1,:]
            sep_hs = result.hidden_states[layer_id][:,-1,:]

            expln_.append(expln)
    
    expln = sum(expln_)
    return pred, expln, str_list

#%%
import matplotlib.pyplot as plt
# test_str = dataset['test']['sentence'][21] #[3,8,13, 14,15, 17, 18, 91]
test_str = "this is a bad movie ."
pred, expln, str_list = explain(model, test_str)
fig = plt.figure(figsize=(4, 2), dpi=80)
ax = fig.add_axes([0,0,1,1])
langs = str_list#[1:]
expln = expln.data[0].numpy()
# expln[expln<0] = 0
students = expln#1:]
ax.bar(langs,students)
plt.xticks(rotation = 60)
plt.show()
print(test_str)
labels = ['negative', 'positive']
print(f"prediction: {labels[pred.item()]}")
# %%
