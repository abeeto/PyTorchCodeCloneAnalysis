import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 32
LR = 0.01
LR_factor = 0.1
EPOCHS = 150

criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LR_factor, patience = 25, verbose = True)
l = None

dataset = abstract_Dataset(data)
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

i = 0
tb = SummaryWriter()
for epoch in tqdm(range(EPOCHS), desc = "epochs", leave = True):
    for images in tqdm(trainloader, desc = "in-epoch", leave = False):
        optimizer.zero_grad()
        dec = net(images)
        loss = criterion(dec, images)
        loss.backward()
        optimizer.step()
        l = loss.item()
        i += 1
        tb.add_scalar("loss", l, i)
    tb.add_figure("decoded images", display_image(), epoch)
    scheduler.step(l)
tb.close()
