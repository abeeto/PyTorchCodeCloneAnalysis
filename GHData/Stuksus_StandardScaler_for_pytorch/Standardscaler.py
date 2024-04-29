# ## Standardscaler for pytorch
import torch


class Standardscaler(torch.nn.Module):
	
    def __init__(self):
        super().__init__()

    def forward(self, input_batch):
        std, mean = torch.std_mean(input_batch.type(torch.float32), unbiased=False)
        total = (input_batch - mean) / std
        return total
