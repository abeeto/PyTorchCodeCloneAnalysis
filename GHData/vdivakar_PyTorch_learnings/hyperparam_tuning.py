'''Use cartesian product instead of multi-nested loops for each HyperParam. See EOF'''
from itertools import product # For Cartesian product

from impl_forward_method import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
torch.set_grad_enabled(True)

net = MyNet()

parameters = dict(
    lr = [0.01, 0.001],
    batch_size = [100, 1000],
    shuffle = [True, False]
)
param_values = [v for v in parameters.values()]

for lr, batch_size, shuffle in product(*param_values):

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)

    summary_name = f'batch_size={batch_size} lr={lr} shuffle={shuffle}'
    tb = SummaryWriter(comment=summary_name)
    tb.add_image('my_images', grid)
    tb.add_graph(net, images)

    total_epocs = 10
    for epoch in range(total_epocs):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch
            preds = net(images)
            loss = F.cross_entropy(preds, labels)
            
            optimizer.zero_grad() # important. Pytorch accumulates grads
            loss.backward()       # Calculate gradients
            optimizer.step()      # Update weights

            total_loss += loss.item() * batch_size # Batch size will be different
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

        tb.add_scalar('dv_Loss', total_loss, epoch)
        tb.add_scalar('Num correct', total_correct, epoch)
        
        for name, weight in net.named_parameters(): #Adding all weights & gradients to TB
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        print(
            "epoch: ", epoch,
            "total_correct: ", total_correct,
            "total_loss: ", total_loss
        )
    tb.close()

'''
for lr, batch_size, shuffle in product(*param_values): 
    print (lr, batch_size, shuffle)

0.01 100 True
0.01 100 False
0.01 1000 True
0.01 1000 False
0.001 100 True
0.001 100 False
0.001 1000 True
0.001 1000 False

> Naming of TB files will be like:
Dec07_23-38-32_Divakars-MacBook-Pro.localbatch_size=100 lr=0.01 shuffle=True
Dec07_23-38-53_Divakars-MacBook-Pro.localbatch_size=100 lr=0.01 shuffle=False
Dec07_23-39-14_Divakars-MacBook-Pro.localbatch_size=1000 lr=0.01 shuffle=True
'''