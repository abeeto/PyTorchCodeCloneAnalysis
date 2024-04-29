from impl_forward_method import *
import torch.optim as optim

'''We disabled graph construction previously.
So, we need to enable this for training / grad calculation'''
torch.set_grad_enabled(True)

net = MyNet()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(net.parameters(), lr=0.01)

batch = next(iter(train_loader))
images, labels = batch

preds = net(images) # Pass batch
loss = F.cross_entropy(preds, labels) # Calculate loss

loss.backward() # Calculate gradients
optimizer.step() # Update weights

print("loss 1: ", loss.item())
preds = net(images)
loss = F.cross_entropy(preds, labels)
print("loss 2: ", loss.item())