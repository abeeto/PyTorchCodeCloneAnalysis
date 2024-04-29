import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from beta_vae_edit import BetaVAE
import read_bvp
import time
import matplotlib.pyplot as plt

USE_EXIST_MODEL = False
TRAIN_ALL = False
ALL_MOTION = [1, 2, 3, 4, 5, 6]
N_MOTION = len(ALL_MOTION)
batch_size = 32
EPOCH = 10
device = "mps"
vae_model = BetaVAE(in_channels=1, latent_dim=16)

vae_model.to(device)


# if USE_EXIST_MODEL == True:
#     model.load_state_dict(torch.load("model_save/train_1666076923.pt"))

if TRAIN_ALL == True:
    print("Train on all data")
    full_dataset = read_bvp.BVPDataSet(data_dir="data/BVP", motion_sel=ALL_MOTION)
else:
    print("Train on test data")
    full_dataset = read_bvp.BVPDataSet(data_dir="testdata/user1", motion_sel=ALL_MOTION)

train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


TIME_STEPS = full_dataset.get_T_max()


# criterion = MSELoss_SEQ()


optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 80], gamma=0.1)
begin_time = int(time.time())


one_batch_data = next(iter(test_loader))[0]
draw_data = one_batch_data[0].permute(0, 2, 3, 1)
for i in range(TIME_STEPS):
    plt.subplot(2, TIME_STEPS, i+1)
    plt.imshow(draw_data[i, :, :, :], cmap="gray")
    plt.axis('off')


for epoch in range(EPOCH):
    #### train ####
    vae_model.train()
    train_loss_sum = 0.0
    # train_sample_sum = 0
    for idx, data in enumerate(train_loader):
        img, label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = vae_model(img)  # [self.decode(z), input, mu, log_var]
        train_batch_loss = vae_model.loss_function(output, M_N=0.005)
        train_batch_loss["loss"].backward()
        optimizer.step()
        train_loss_sum += train_batch_loss["loss"].item()
    train_epoch_loss = train_loss_sum / (idx+1)

    print(f"Epoch [{epoch}] lr: {optimizer.state_dict()['param_groups'][0]['lr']}", end=" ")
    print('Train epoch loss:', train_epoch_loss)
    scheduler.step()

#     #### validation ####
#     best_loss = 10000.0
#     encoder.eval()
#     decoder.eval
#     size = len(test_loader.dataset)  # type: ignore
#     num_batches = len(test_loader)
#     test_loss = 0
#     with torch.no_grad():
#         for img, label in test_loader:
#             img, label = img.to(device), label.to(device)
#             output_encode = encoder(img)
#             output_decode = decoder(output_encode)
#             test_loss += criterion(img, output_decode).item()

#     test_loss /= num_batches

#     # if (test_loss < best_loss) & (epoch > 20):
#     #     best_loss = test_loss
#     #     torch.save(model.state_dict(), f"model_save/train_{begin_time}.pt")


draw_data_encode = vae_model(one_batch_data.to(device))

draw_data = draw_data_encode[0][0].cpu().permute(0, 2, 3, 1).detach().numpy()

for i in range(TIME_STEPS):
    plt.subplot(2, TIME_STEPS, TIME_STEPS+i+1)
    plt.imshow(draw_data[i, :, :, :], cmap="gray")
    plt.axis('off')
plt.show()
