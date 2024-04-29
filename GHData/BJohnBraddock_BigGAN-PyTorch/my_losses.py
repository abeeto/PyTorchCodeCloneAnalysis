import torch
import torch.nn.functional as F

# Pleasant : Maximize vca_fake = 1
# # Loss is lowest when -1 * 1 = -1, highest when -1 * 0
def loss_vca_gen_pleasant(vca_fake):
  loss = -torch.mean(vca_fake)
  return loss

# Unpleasant: Maximize vca_fake = 0
def loss_vca_gen_unpleasant(vca_fake):
  loss = torch.mean(vca_fake)
  return loss

generator_vca_loss_pleasant = loss_vca_gen_pleasant
generator_vca_loss_unpleasant = loss_vca_gen_unpleasant