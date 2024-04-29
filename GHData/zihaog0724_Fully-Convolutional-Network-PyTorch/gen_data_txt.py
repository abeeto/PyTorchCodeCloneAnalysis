import os
import random

imgs = os.listdir("./data/img")
random.shuffle(imgs)

split = int(0.0 * len(imgs))
train_imgs = imgs[:split]
val_imgs = imgs[split:]

train_txt = open("./train.txt", "w")
val_txt = open("val.txt", "w")

for i in train_imgs:
    train_txt.write(i.strip(".jpg") + "\n")

for j in val_imgs:
    val_txt.write(j.strip(".jpg") + "\n")
    

