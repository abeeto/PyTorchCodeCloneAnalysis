import config.ensemble_config as ensemble_config
import torch
import numpy as np
from tqdm import tqdm
from dataset import test_dataloader
from model import PhoBERTLSTM_base, PhoBERTLSTM_large, PhoBertFeedForward_base, PhoBertFeedForward_large
from trainer import PhoBERTModel
from torchmetrics import Accuracy, F1Score
from termcolor import colored

def GetModel(model_name, ckpt, number):
    if model_name == "FeedForward-base":
        model = PhoBertFeedForward_base(from_pretrained=False)
        print(colored(f"Model {number}: PhoBERT FeedForward base", "green"))
    elif model_name == "FeedForward-large":
        model = PhoBertFeedForward_large(from_pretrained=False)
        print(colored(f"Model {number}: PhoBERT FeedForward large", "green"))
    elif model_name == "LSTM-base":
        model = PhoBERTLSTM_base(from_pretrained=False)
        print(colored(f"Model {number}: PhoBERT LSTM base", "green"))
    elif model_name == "LSTM-large":
        model = PhoBERTLSTM_large(from_pretrained=False)
        print(colored(f"Model {number}: PhoBERT LSTM large", "green"))
    system = PhoBERTModel(model)
    system.load_state_dict(ckpt["state_dict"])
    system.eval()
    print(colored(f"MODEL {number} LOADED!", "red"))
    return system.to(ensemble_config.DEVICE)

def ensemble_fn(pred1, pred2):
    return pred1 * 0.8 + pred2 * 0.2

if __name__ == '__main__':
    # Metrics
    accuracy = Accuracy().to(ensemble_config.DEVICE)
    f1_score = F1Score().to(ensemble_config.DEVICE)

    # Checkpoint
    ckpt1 = torch.load(ensemble_config.CKPT1, map_location=ensemble_config.DEVICE)
    ckpt2 = torch.load(ensemble_config.CKPT2, map_location=ensemble_config.DEVICE)

    # Get Model
    model1 = GetModel(ensemble_config.MODEL1, ckpt1, 1)
    model2 = GetModel(ensemble_config.MODEL2, ckpt2, 2)

    # Evaluate on Test Set
    acc_list = []
    f1_list = []
    with torch.no_grad():
        loop = tqdm(test_dataloader)
        for data in loop:
            pred1 = model1(data["input_ids"].to(ensemble_config.DEVICE), data["attention_mask"].to(ensemble_config.DEVICE))
            pred2 = model2(data["input_ids"].to(ensemble_config.DEVICE), data["attention_mask"].to(ensemble_config.DEVICE))

            prediction = ensemble_fn(pred1, pred2)

            acc = accuracy(prediction, data["sentiment"].to(ensemble_config.DEVICE))
            f1 = f1_score(prediction, data["sentiment"].to(ensemble_config.DEVICE))
            
            acc_list.append(acc.item())
            f1_list.append(f1.item())

            loop.set_description("Proceeding")
            loop.set_postfix(acc=acc.item())

    print(colored(f"Accuracy: {np.mean(np.array(acc_list)):.3f}", "blue"))
    print(colored(f"F1 Score: {np.mean(np.array(f1_list)):.3f}", "blue"))