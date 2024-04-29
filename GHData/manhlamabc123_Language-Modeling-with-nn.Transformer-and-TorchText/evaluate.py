from torch import nn, Tensor
import torch
from transformer_model import generate_square_subsequent_mask
from constants import *
from data_process import get_batch

def evaluate(model: nn.Module, eval_data: Tensor, n_tokens, criterion) -> float:
    model.eval()
    total_loss = 0
    src_mask = generate_square_subsequent_mask(BPTT).to(DEVICE)
    with torch.no_grad():
        for i in range(0, eval_data.size() - 1, BPTT):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != BPTT:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, n_tokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)