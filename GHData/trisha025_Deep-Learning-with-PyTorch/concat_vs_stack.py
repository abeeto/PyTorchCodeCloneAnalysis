import torch
t1 = torch.tensor([1,1,1])
print(t1.unsqueeze(dim=0).shape)
print(t1.unsqueeze(dim=1).shape)
print(t1.shape)

t2 = torch.tensor([2,2,2])
t3 = torch.tensor([3,3,3])

t_cat = torch.cat(
    (t1,t2,t3), dim=0
)

print(t_cat)

t_stack = torch.stack(
    (t1,t2,t3)
)

print(t_stack)