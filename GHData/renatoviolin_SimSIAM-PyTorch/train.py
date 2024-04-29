# %%
try:
    get_ipython()
    NUM_WORKERS = 0
except:
    NUM_WORKERS = 8
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint", required=False)
    # args = parser.parse_args()


from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from models import SiamModel
import dataset
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
CKPT = 'siam.pt'
EPOCHS = 20


# %%
data = glob.glob('../data_cat_dog/no_label/*.jpg')
train_dataset = dataset.SiamDataset(data)
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)


# %%
siam_model = SiamModel().to(DEVICE)
# if args.checkpoint:
#     print('reloading checkpoint....')
#     siam_model = torch.load(args.checkpoint, map_location=DEVICE)

optimizer = torch.optim.Adam(siam_model.parameters(), lr=5e-4)

total_loss = []
for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}')
    t_loss = []
    t_loader = tqdm(train_dataloader)
    for batch in t_loader:
        optimizer.zero_grad()
        img_1, img_2 = batch[0].to(DEVICE), batch[1].to(DEVICE)
        loss = siam_model.train(img_1, img_2)
        loss.backward()
        optimizer.step()

        _total_loss = loss.detach().cpu().numpy().item()

        t_loss.append(_total_loss)
        t_loader.set_description(f'total loss: {_total_loss:.4f}')
        total_loss.append(np.mean(t_loss))

    print(f"train loss: {np.mean(t_loss):.4f}")

    if (epoch % 2) == 0:
        torch.save(siam_model, CKPT)

torch.save(siam_model, CKPT)

x = np.arange(len(total_loss))
plt.title('SimSiam training loss')
plt.plot(x, total_loss)
plt.savefig('siam.jpg')
