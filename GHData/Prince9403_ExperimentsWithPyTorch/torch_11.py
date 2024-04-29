import torch
import torch.nn as nn

lin_layer = nn.Linear(in_features=1, out_features=1)

optimizer = torch.optim.Adam(lin_layer.parameters(), lr=0.01)

x0 = 7 * torch.ones(1)
x0 = x0.view(1, 1, 1)

x1 = lin_layer(x0)
x1.backward()
optimizer.step()

"""
We do not get any error as we build the computation graph for the second time.
If we omit the line "x1 = lin_layer(x0)" then we get the error "Trying to backward through the graph a second time"
"""
x1 = lin_layer(x0)
x1.backward()
optimizer.step()

