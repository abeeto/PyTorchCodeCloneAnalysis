import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#I use this to collect all the results from the games played, and concatenate them into one numpy array, which is then also saved as a file
A =  np.array([[0,0,0]])
B = np.array([[0,0]])
mode = input("To create a batch input 'b', to show visualize batch data input 'v\n")
while True:
    if mode == 'b':
        for i in range(100):
            name = 'iteration' + str(i) + '.npy'
            with open(name, 'rb') as f:
                a = np.load(f)
                b = np.load(f)
                A = np.concatenate((A,a))
                B = np.concatenate((B,b))
        txt = "batch.npy"
        with open(txt, 'wb') as f:
            np.save(f,A) 
            np.save(f, B)
        mode = "x"
    elif mode == 'v':
        name = "batch.npy"
        with open(name, 'rb') as f:
                a = np.load(f)
                b = np.load(f)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(len(b)):
            if b[i][0] == 1:
                m = 'x'
                clr = 'red'
            else:
                m = 'o'
                clr = 'green'
            ax.scatter3D(a[i,0], a[i,1], a[i,2], color = clr, marker = m)
        ax.set_xlabel('Enemy Aggresiveness', fontweight ='bold')
        ax.set_ylabel('Player Health', fontweight ='bold')
        ax.set_zlabel('Player Power', fontweight ='bold')
        plt.show()
        mode = "x"

    else:
        mode = input("To create a batch input 'b', to show visualize batch data input 'v")



