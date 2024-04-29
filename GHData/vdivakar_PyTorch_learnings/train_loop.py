from impl_forward_method import *
import torch.optim as optim

'''We disabled graph construction previously.
So, we need to enable this for training / grad calculation'''
torch.set_grad_enabled(True)

net = MyNet()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
print("Training Set: ", len(train_set))
optimizer = optim.Adam(net.parameters(), lr=0.01)

total_epocs = 10

for epoch in range(total_epocs):
    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch
        preds = net(images)
        loss = F.cross_entropy(preds, labels)
        
        optimizer.zero_grad() # important. Pytorch accumulates grads
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += preds.argmax(dim=1).eq(labels).sum().item()

    print(
        "epoch: ", epoch,
        "total_correct: ", total_correct,
        "total_loss: ", total_loss
    )