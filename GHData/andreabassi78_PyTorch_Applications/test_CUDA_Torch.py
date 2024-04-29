import torch
# setting device on GPU if available, else CPU

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')


print('Using device:', device, '\n')

# torch.rand(1).to(device)
# torch.rand(1, device=device)

t1 = torch.randn(1,1)
t2 = torch.randn(1,2).to(device)

#print (t1)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
torch.cuda.empty_cache()
