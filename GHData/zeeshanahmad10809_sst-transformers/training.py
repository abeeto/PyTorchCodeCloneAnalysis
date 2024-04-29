import torch
from sst_transformers.dataset import SSTDataset
from torch.utils.data import DataLoader
from sst_transformers.models import load_transformer
from sst_transformers.utils import transformer_params
from utils import evaluation_metrics, save_model, root_and_binary_title
from tqdm import tqdm
from math import ceil
from loguru import logger
import numpy as np
import os
import time
from datetime import timedelta

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_step(model, inputs, labels, optimizer):
    optimizer.zero_grad()

    labels = labels.unsqueeze(0)
    outputs = model(
        inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels
    )
    loss, logits = outputs[:2]

    loss.backward()
    optimizer.step()

    return logits, loss


def eval_step(model, inputs, labels):
    labels = labels.unsqueeze(0)
    outputs = model(
        inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels
    )
    loss, logits = outputs[:2]

    return logits, loss


def train_epoch(model, tokenizer, train_dataset, optimizer, batch_size):
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    correct_count = 0
    total_loss = 0

    model.train()
    with tqdm(
        total=ceil(len(train_dataset) / batch_size), desc="train", unit="batch"
    ) as pbar:
        for text, sentiment in train_loader:
            text = tokenizer(text, padding=True, return_tensors="pt").to(device)
            sentiment = sentiment.to(device)

            logits, loss = train_step(model, text, sentiment, optimizer)

            preds = torch.argmax(logits, axis=1)
            correct_count += (preds == sentiment).sum().item()
            total_loss += loss.item()
            pbar.update(1)

    return correct_count / len(train_dataset), total_loss / len(train_dataset)


def eval_epoch(model, tokenizer, eval_dataset, batch_size, split):
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True)

    correct_count = 0
    total_loss = 0
    y_pred = list()
    y_true = list()

    model.eval()
    with torch.no_grad():
        with tqdm(
            total=ceil(len(eval_dataset) / batch_size), desc=split, unit="batch"
        ) as pbar:
            for text, sentiment in eval_loader:
                text = tokenizer(text, padding=True, return_tensors="pt").to(device)
                sentiment = sentiment.to(device)

                logits, loss = eval_step(model, text, sentiment)

                preds = torch.argmax(logits, axis=1)
                y_pred += preds.cpu().numpy().tolist()
                y_true += sentiment.cpu().numpy().tolist()

                correct_count += (preds == sentiment).sum().item()
                total_loss += loss.item()
                pbar.update(1)

    metrics_score = evaluation_metrics(y_true, y_pred, split=split)
    return (
        correct_count / len(eval_dataset),
        total_loss / len(eval_dataset),
        metrics_score,
    )


def train(name, root, binary, epochs=25, patience=3, save=False):

    # load model and tokenizer..
    try:
        transformer_container = load_transformer(name, binary)
    except ValueError:
        logger.error("Invalid transformer name!")
        os._exit(0)
    model = transformer_container["model"]
    model = model.to(device)
    tokenizer = transformer_container["tokenizer"]

    # load batch_size and learning rate..
    params_container = transformer_params(name)
    batch_size = params_container["batch_size"]
    learning_rate = params_container["learning_rate"]

    # load train, dev and test datasets..
    train_dataset = SSTDataset(root=root, binary=binary, split="train")
    dev_dataset = SSTDataset(root=root, binary=binary, split="dev")
    test_dataset = SSTDataset(root=root, binary=binary, split="test")

    # Intialize optimizer..
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize training variables..
    best_acc = 0.0
    best_loss = np.inf
    stopping_step = 0
    best_model_name = None

    total_train_seconds = 0
    for epoch in range(epochs):

        start = time.time()
        train_acc, train_loss = train_epoch(
            model, tokenizer, train_dataset, optimizer, batch_size
        )
        end = time.time()
        total_train_seconds += end - start
        logger.info(
            f"epoch: {epoch+1}, transformer: {name}, train_loss: {train_loss:.4f}, train_acc: {train_acc*100:.2f}"
        )

        dev_acc, dev_loss, _ = eval_epoch(
            model, tokenizer, dev_dataset, batch_size, "dev"
        )
        logger.info(
            f"epoch: {epoch+1}, transformer: {name}, dev_loss: {dev_loss:.4f}, dev_acc: {dev_acc*100:.2f}"
        )

        test_acc, test_loss, test_evaluation_metrics = eval_epoch(
            model, tokenizer, test_dataset, batch_size, "test"
        )
        logger.info(
            f"epoch: {epoch+1}, transformer: {name}, test_loss: {test_loss:.4f}, test_acc: {test_acc*100:.2f}"
        )
        logger.info(
            f"epoch: {epoch+1}, transformer: {name}, "
            f"test_precision: {test_evaluation_metrics['test_precision']*100:.2f}, "
            f"test_recall: {test_evaluation_metrics['test_recall']*100:.2f}, "
            f"test_f1_score: {test_evaluation_metrics['test_f1_score']*100:.2f}, "
            f"test_accuracy_score: {test_evaluation_metrics['test_accuracy']*100:.2f}"
        )
        logger.info(
            f"epoch: {epoch+1}, transformer: {name}, test_confusion_matrix: \n"
            f"{test_evaluation_metrics['test_confusion_matrix']}"
        )

        logger.info(
            f"Total training time elapsed: {timedelta(seconds=total_train_seconds)}"
        )
        logger.info(
            f"Mean time per train epoch: {timedelta(seconds=total_train_seconds/(epoch+1))}"
        )

        # save best model and delete previous ones...
        if save:
            if test_acc > best_acc:
                best_acc = test_acc
                phrase_type, label = root_and_binary_title(root, binary)
                model_name = "{}_{}_{}_{}.pickle".format(
                    name, phrase_type, label, epoch
                )
                save_model(model, model_name, best_model_name)

        # Implement early stopping here
        if test_loss < best_loss:
            best_loss = test_loss
            stopping_step = 0
        else:
            stopping_step += 1

        if stopping_step >= patience:
            logger.info("EarlyStopping!")
            os._exit(1)
