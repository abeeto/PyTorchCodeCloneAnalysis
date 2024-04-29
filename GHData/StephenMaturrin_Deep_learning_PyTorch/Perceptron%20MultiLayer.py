
from __future__ import print_function
from __future__ import division
import  functions
import gzip # pour décompresser les données
import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import pickle  # pour désérialiser les données
# fonction qui va afficher l'image située à l'index index
import  variables as var
import matplotlib.patches as mpatches



if __name__ == '__main__':
    # nombre d'image lues à chaque fois dans la base d'apprentissage (laisser à 1 sauf pour la question optionnelle sur les minibatchs)
    TRAIN_BATCH_SIZE = 1


    with gzip.open('mnist.pkl.gz','rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # print(p)
    # on charge les données de la base MNIST
    # data = pickle.load(gzip.open('mnist.pkl.gz'))
    # images de la base d'apprentissage [torch.FloatTensor of size 63000x784]
    train_data = torch.DoubleTensor(data[0][0])


    # labels de la base d'apprentissage [torch.FloatTensor of size 63000x784]
    train_data_label = torch.DoubleTensor(data[0][1])


    # images de la base de test [torch.FloatTensor of size 7000x784]
    test_data = torch.DoubleTensor(data[1][0])



    # labels de la base de test [torch.FloatTensor of size 7000x10]
    test_data_label = torch.DoubleTensor(data[1][1])

    # on crée la base de données d'apprentissage (pour torch)
    train_dataset = torch.utils.data.TensorDataset(train_data,train_data_label)
    # on crée la base de données de test (pour torch)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_data_label)
    # on crée le lecteur de la base de données d'apprentissage (pour torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # on crée le lecteur de la base de données de test (pour torch)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    # 10 fois



N_Hidden_layer = 300
X = torch.DoubleTensor(1, var.N_FEATURES + 1)  # R 1*785

W_Entry_Layer = torch.rand(var.N_FEATURES+1,N_Hidden_layer).uniform_(-0.1,0.1) # R 785*30

W_Hidden_Layer = torch.rand(N_Hidden_layer+1,var.N_CLASSES).uniform_(-1,1) # R 30*10
Y = torch.DoubleTensor(1, var.N_CLASSES)      # R 1*10
y1 = torch.DoubleTensor(1, 31)
y2 = torch.DoubleTensor(1, 10)

bias = torch.DoubleTensor([1])

label = torch.DoubleTensor(1, var.N_CLASSES)

prediction = torch.DoubleTensor(1,var.N_CLASSES)

deltaLabel = torch.DoubleTensor(1,var.N_CLASSES)

aux = torch.DoubleTensor(var.N_FEATURES+1, 1)

deltaW  = torch.DoubleTensor(var.N_FEATURES+1,var.N_CLASSES)



sigmoid_v = numpy.vectorize(functions.sigmoid)
DFsigmoid_v = numpy.vectorize(functions.DFsigmoid)
N_Hidden_layer = 30

y = [[] for _ in range(3)]
for j in range(1,2):

    N_Hidden_layer = j
    print(N_Hidden_layer)
    W_Entry_Layer = torch.rand(var.N_FEATURES + 1, N_Hidden_layer).uniform_(-0.1, 0.1)  # R 785*30

    W_Hidden_Layer = torch.rand(N_Hidden_layer + 1, var.N_CLASSES).uniform_(-1, 1)  # R 30*10


    y[0].append(N_Hidden_layer)
    for i in range(1):

        for i in range(var.N_IMAGES_TRAIN):
            #########################################################################

            X = torch.cat((bias,train_data[i, :]), 0)
            X= X [numpy.newaxis]
            label =   train_data_label [i, :]
            Z2 = torch.mm(X,W_Entry_Layer.double()) #R 1*30 = R 1*785 . R 785*30
            a2= torch.sigmoid(Z2)
            a2 = a2.view(1, N_Hidden_layer)
            a2 = torch.cat((bias,a2[0,:]),0).view(1,N_Hidden_layer+1)  #R 1*31 a2[0, :
            Z3= torch.mm(a2,W_Hidden_Layer.double())# R 1*10 = R 1*31 R 31*10
            ################################################################################
            delta2 = torch.add(label.view(1,var.N_CLASSES),Z3*-1)  #R 1*10
            aux = torch.sigmoid(Z2) * (1-torch.sigmoid(Z2))
            delta1 = torch.mm(delta2,numpy.transpose((W_Hidden_Layer[1:,:].double()))) # R 1* 30 = R 1*10 . R 10*31
            delta1= torch.mul(delta1,aux) #delta2[0,1:31]
            delta_W_Entry_Layer = var.EPSILON * torch.mm(numpy.transpose((X)),delta1)
            W_Entry_Layer = torch.add(delta_W_Entry_Layer,W_Entry_Layer.double())
            delta_W_Hidden_Layer =  var.EPSILON * torch.mm(torch.t(Z2),delta2)    #R 1*30 #R 1*10
            W_Hidden_Layer[1:,:] = torch.add(delta_W_Hidden_Layer,W_Hidden_Layer[1:,:].double())#  W_Hidden_Layer[1,:]


    accurracy= 0
    for i in range(var.N_IMAGES_TEST):
        X = torch.cat((bias, test_data[i, :]), 0)
        X = X[numpy.newaxis]

        label = test_data_label[i, :]
        Z2 = numpy.dot(X, W_Entry_Layer)  # R 1*30 = R 1*785 . R 785*30
        a2 = sigmoid_v(Z2)  # R 1*31
        a2 = numpy.concatenate((bias, a2[0, :]), 0)  # R 1*31 a2[0, :]
        # a2 = numpy.concatenate((bias, a2[0, :]), 0)  # R 1*31
        a2 = a2[numpy.newaxis]
        a2 = torch.from_numpy(a2)

        Z3 = numpy.dot(a2, W_Hidden_Layer)  # R 1*10 = R 1*31 R 31*10
        Z3 = torch.from_numpy(Z3)

        # print("predicted %f label %f" % (numpy.argmax(Z3), numpy.argmax(label)))
        if (numpy.argmax(Z3) == numpy.argmax(label)):
            accurracy += 1

    y[1].append(accurracy / var.N_IMAGES_TEST * 100)
    print("Valeurs bien predit: %d " % (accurracy))
    print("Valeurs mal predit:  %d " % (var.N_IMAGES_TEST))
    print("Taux de reussite:    %f" % (accurracy/var.N_IMAGES_TEST*100))
    print("Taux d'erreur:       %f" %  (100-(accurracy/var.N_IMAGES_TEST*100)))
    #

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.grid(True)
gridlines = ax0.get_xgridlines() + ax0.get_ygridlines()
plt.yscale('linear')
plt.xscale('linear')
for line in gridlines:
    line.set_linestyle('-.')
plt.plot(y[0], y[1], 'bs',y[0], y[1])
plt.ylabel('Accuracy')
plt.xlabel('Neurones')
blue_patch = mpatches.Patch(color='blue', label='Accuracy')
plt.legend(handles=[blue_patch])
plt.show()