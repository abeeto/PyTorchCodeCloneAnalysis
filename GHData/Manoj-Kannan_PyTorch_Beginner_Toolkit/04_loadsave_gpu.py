import torch
import torch.nn as nn

# Save on GPU, Load on GPU
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), path)

model = Model(*args, **kwargs)
torch.load_state_dict(torch.load(path))
model.to(device)

# Save on CPU, Load on GPU
torch.save(model.state_dict(), path)

device = torch.device('cuda')
model = Model(*args, **kwargs)
torch.load_state_dict(torch.load(path, map_location=device))
model.to(device)

# Save on GPU, Load on CPU
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), path)

device = torch.device('cpu')
model.to(device)
torch.load_state_dict(torch.load(path, map_location=device))
