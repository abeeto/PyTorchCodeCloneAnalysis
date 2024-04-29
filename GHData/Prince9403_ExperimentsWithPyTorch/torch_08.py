import torch
import torch.nn as nn

rnn = nn.RNN(input_size=1, hidden_size=2, num_layers=1, batch_first=False)
lin_layer = nn.Linear(in_features=2, out_features=1)

h = torch.randn(1, 1, 2)

x0 = 7 * torch.ones(1)
x0 = x0.view(1, 1, 1)

print("x0 size:", x0.size())
print("h size:", h.size())

optimizer = torch.optim.Adam(list(rnn.parameters()) + list(lin_layer.parameters()), lr=0.01)

for i in range(2):
    optimizer.zero_grad()
    out, h = rnn(x0, h)
    print("out =" ,out)
    print("h =", h)
    out = out.view(-1, 2)
    out = lin_layer(out)
    out.backward()
    """
    Error (when i = 1): Trying to backward through the graph a second time.
    The reason is that when i = 1, we take h from the previous iteration,
    and this h is already a vertex in a computation graph. So, we
    try to propagate through this graph (the subgraph of the whole graph)
    for the second time. This causes the error. 
    """
    optimizer.step()

