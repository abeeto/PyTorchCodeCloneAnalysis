from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from covid import show
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import random

def show(p_img, n_img, name, norm):
	# convert tensor to numpy array
    p_npimg = p_img.numpy()
    n_npimg = n_img.numpy()
    if norm:
        p_npimg = 0.5146469 + 0.3062062 * p_npimg
        n_npimg = 0.5146469 + 0.3062062 * n_npimg
	# Convert to H*W*C shape
    p_npimg_tr=np.transpose(p_npimg, (1,2,0))
    n_npimg_tr=np.transpose(n_npimg, (1,2,0))
    plt.subplot(1,2,1)
    plt.imshow(p_npimg_tr)
    plt.title("COVID-19")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(n_npimg_tr)
    plt.title("Non COVID-19")
    plt.axis('off')
    plt.savefig(name)
    plt.clf()

plt.rcParams["figure.figsize"] = (16, 16)
temp_transformer = transforms.Compose([transforms.ToTensor()])
covid_ds = ImageFolder("./full_data", temp_transformer)

sample_size = 4
pos_len = 2
neg_len = 2
neg_sample = []
pos_sample = []
neg = []
pos = []
idx_list = list(range(len(covid_ds)))
random.shuffle(idx_list)

for i in idx_list:
    idx = covid_ds.targets[i]
    if idx == 0:
        neg_len -= 1
        neg_sample.append(i)
    if neg_len == 0:
        break
for i in neg_sample:
    neg.append(covid_ds[i][0])

for i in idx_list:
    idx = covid_ds.targets[i]
    if idx == 1:
        pos_len -= 1
        pos_sample.append(i)
    if pos_len == 0:
        break
for i in pos_sample:
    pos.append(covid_ds[i][0])


pos_sample = make_grid(pos, nrow=2, padding=3)
neg_sample = make_grid(neg, nrow=2, padding=3)
show(pos_sample, neg_sample, "grid.png", True)
