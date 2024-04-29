from impl_forward_method import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
torch.set_grad_enabled(True)

net = MyNet()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
print("Training Set: ", len(train_set))
optimizer = optim.Adam(net.parameters(), lr=0.01)


images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb = SummaryWriter()
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

        total_loss += loss.item()
        total_correct += preds.argmax(dim=1).eq(labels).sum().item()

    tb.add_scalar('dv_Loss', total_loss, epoch)
    tb.add_scalar('Num correct', total_correct, epoch)
    tb.add_histogram("conv1.bias", net.conv1.bias, epoch)
    tb.add_histogram("conv1.weight", net.conv1.weight, epoch)
    tb.add_histogram("conv1.weight.grad", net.conv1.weight.grad, epoch)

    print(
        "epoch: ", epoch,
        "total_correct: ", total_correct,
        "total_loss: ", total_loss
    )
tb.close()