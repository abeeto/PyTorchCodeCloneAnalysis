import numpy as np
import torch
import matplotlib.pyplot as plt

anc_grid = 4
k = 1

anc_offset = 1/(anc_grid*2)

z= np.linspace(anc_offset, 1-anc_offset, anc_grid)

anc_x = np.repeat(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)
anc_y = np.tile(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)

anc_ctrs = np.tile(np.stack([anc_x,anc_y], axis=1), (k,1))
anc_sizes = np.array([[1/anc_grid,1/anc_grid] for i in range(anc_grid*anc_grid)])
anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1), requires_grad=False).float()

grid_sizes = torch.tensor(np.array([1/anc_grid]), requires_grad=False).unsqueeze(1)

# plt.scatter(anc_x, anc_y,marker='+')
#
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.show()

def hw2corners(ctr, hw):
    return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)
anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])


from matplotlib import patches, patheffects

# Create figure and axes
fig,ax = plt.subplots(1)

# Create a Rectangle patch

# Add the patch to the Axes
# ax.add_patch(rect)
#
# plt.show()
# def draw_outline(o, lw):
#     o.set_path_effects([patheffects.Stroke(
#         linewidth=lw, foreground='black'), patheffects.Normal()])
#
# def draw_rect(ax, b, color='white'):
#     patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
#     draw_outline(patch, 4)

for anchor in anchor_cnr:
    rect = patches.Rectangle(anchor[:2], anchor[2],  anchor[3], linewidth=1,fill=False, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()