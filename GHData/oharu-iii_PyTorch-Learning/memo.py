import torch

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.ones(1, device=device)
# model.to(device)
print(device)