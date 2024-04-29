import torch
import torch.distributed

print("CUDA available:", torch.cuda.is_available())
print("Distributed available:", torch.distributed.is_available())
print("NCCL (distributed GPU) available:", torch.distributed.is_nccl_available())
