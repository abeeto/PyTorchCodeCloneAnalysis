import os
import sys

os.environ["TOKENIZERS_PARALLELISM"]="false"
import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from absl import app, flags
from accelerate import Accelerator
from datasets import load_dataset
from ml_collections import config_flags
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# custom modules
from src.trainer import Trainer, TrainerArguments
from src.utils import set_seed
config_flags.DEFINE_config_file(
    "config",
    default=None,
    help_string="Training Configuration from `configs` directory",
)
flags.DEFINE_bool("wandb_enabled", default=True, help="enable Weights & Biases logging")

FLAGS = flags.FLAGS
FLAGS(sys.argv)   # need to explicitly to tell flags library to parse argv before you can access FLAGS.xxx
cfg = FLAGS.config
wandb_enabled = FLAGS.wandb_enabled

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_labels = 2
        self.model =AutoModelForSequenceClassification.from_pretrained(cfg.model_ckpt, num_labels=self.num_labels)
        
    def forward(self,input_ids, token_type_ids, attention_mask, label=None):
        out = self.model(input_ids, attention_mask,token_type_ids, labels=label)
        logits = out.logits
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            return logits, loss
        return logits

def main():
    set_seed(42)

    # init 🤗 accelerator
    accelerator = Accelerator(
            device_placement=True,
            step_scheduler_with_optimizer=False,
            mixed_precision=cfg.trainer_args['mixed_precision'],
            gradient_accumulation_steps=cfg.trainer_args['gradient_accumulation_steps'],
            log_with="wandb" if wandb_enabled else None,
        )
    if wandb_enabled:
        # init wandb
        accelerator.init_trackers(project_name="Custom_Accelerate_Trainer_Tests", config=cfg.to_dict())
    
    # disable logging on all processes except main
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # load dataset, model, tokenizer
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckpt)
    model = Model()

    # tokenize and prepare dataset
    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128, return_token_type_ids=True)
        return outputs
    
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["idx", "sentence1", "sentence2"])

    accelerator.print(tokenized_datasets['train'].features)
    tokenized_datasets.set_format("torch")

    # get optimizer and split parameter groups for weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

    # create dataloaders
    def create_dataloaders(train_batch_size=8, eval_batch_size=32):
        train_dataloader = DataLoader(
            tokenized_datasets['train'], shuffle=True, batch_size=train_batch_size
        )
        eval_dataloader = DataLoader(
            tokenized_datasets['validation'],shuffle=False, batch_size=eval_batch_size
        )
        test_dataloader = DataLoader(
            tokenized_datasets['test'],shuffle=False, batch_size=eval_batch_size
        )
        return train_dataloader, eval_dataloader, test_dataloader

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_batch_size=cfg.train_batch_size, eval_batch_size=cfg.val_batch_size)

    # implement custom metrics function
    # these metrics will be calculated and logged after every evaluation phase
    # do all the post-processing on logits in this function
    def compute_metrics(logits, labels):
        preds = np.argmax(logits, axis=1)
        acc_score = accuracy_score(labels, preds)
        f1_sc = f1_score(labels, preds) 
        return {
            "accuracy": acc_score,
            "f1": f1_sc
        }

    # trainer args
    args = TrainerArguments(**cfg.trainer_args)

    # initialize my custom trainer
    trainer = Trainer(
        model=model,
        args=args,
        optimizer=optimizer,
        accelerator=accelerator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        compute_metrics=compute_metrics,
    )

    # call .fit to perform training + validation
    trainer.fit()

    # predict on val and test set at the end of training with the best model
    preds = trainer.predict("./outputs/best_model.bin", test_dataloader)
    val_preds = trainer.predict("./outputs/best_model.bin", val_dataloader)
    val_metrics = compute_metrics(val_preds, tokenized_datasets["validation"]["label"])

    accelerator.print(f"val metrics: {val_metrics}")
    metrics = compute_metrics(preds, tokenized_datasets["test"]["label"])
    accelerator.print(f"Test metrics: {metrics}")
    
    if wandb_enabled:
        # end trackers
        accelerator.end_training()

if __name__ == "__main__":
    main()
