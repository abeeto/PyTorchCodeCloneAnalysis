# correct label 1, fake label 0
# one-hot vector concat dataで
import os
import random
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import model
import hoge
from config import *

interrupt_flag = False # restart interrupted training

# create required directory
required_dirs = ["param", "result", "mnist"]
hoge.make_dir(required_dirs)

mnist_data = MNIST('./mnist/', train=True, download=True,
        transform = transforms.ToTensor())
dataloader = DataLoader(mnist_data, batch_size=mini_batch_num, shuffle=True)

print("\n")
if interrupt_flag:  # restart training
    f = open("./param/tmp.pickle", mode="rb")
    init_epoch = pickle.load(f)
    model = model.GAN(dataloader, interrupting=True)
else:
    init_epoch = 1
    model = model.GAN(dataloader)

# iteration
for epoch in range(init_epoch, epochs+1):
    print("Epoch[%d/%d]:"%(epoch, epochs))
    model.study(epoch)
    model.evaluate()
    model.save_tmp_weight(epoch)
    model.eval_pic(epoch)
model.output()
