
import numpy as np
from AGNews_classification_fun import run

IRs = np.array([10,50,100])
reps = np.array([0,1,2,3,4])
ORs = np.array([True,False])
Methods = np.array(['Original', 'ROS', 'GAN', 'ImbGAN'])

np.random.shuffle(reps)
np.random.shuffle(IRs)
np.random.shuffle(ORs)
np.random.shuffle(Methods)

for rep in reps:
    for IR in IRs:
        for OR in ORs:
            for Method in Methods:
                run(IR=IR,rep=rep,GPU_NUM=3,OR=OR,Method=Method)