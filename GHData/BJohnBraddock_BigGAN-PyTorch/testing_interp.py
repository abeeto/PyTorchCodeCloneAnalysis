import utils
import torch

embed = torch.nn.Embedding(1000, 128)

num_per_sheet = 16
num_midpoints=8
num_classes = 1000

def interp(x0, x1, num_midpoints):
  lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cpu').to(x0.dtype)
  return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))

ys = interp(embed(utils.sample_1hot(num_per_sheet, num_classes, device='cpu')).view(num_per_sheet, 1, -1),
                    embed(utils.sample_1hot(num_per_sheet, num_classes, device='cpu')).view(num_per_sheet, 1, -1),
                    num_midpoints).view(num_per_sheet *(num_midpoints + 2), -1)


# print(ys[0])
print(torch.diff(ys)[0])
print(embed(utils.sample_1hot(1, 1000, device='cpu')).size())