import os
for i in range(0,10,1):
    os.system("python train.py --exp_name Test"+str(i)+"_ex6 --checkpoint checkpoint/Test"+str(i)+"_ex6"+" --dataroot /home/itm1/seungjun/data/MNIST --batch 512 --protocol 2 --isize 61 --latent 32 --epochs 300 --lr 0.0001 --anomaly_class "+str(i))
