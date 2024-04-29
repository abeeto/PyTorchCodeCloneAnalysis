import  functions
import gzip # pour décompresser les données
import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import matplotlib.patches as mpatches
import pickle  # pour désérialiser les données
# fonction qui va afficher l'image située à l'index index
import math
import  variables as var



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
    train_data = torch.Tensor(data[0][0])


    # labels de la base d'apprentissage [torch.FloatTensor of size 63000x784]
    train_data_label = torch.Tensor(data[0][1])


    # images de la base de test [torch.FloatTensor of size 7000x784]
    test_data = torch.Tensor(data[1][0])



    # labels de la base de test [torch.FloatTensor of size 7000x10]
    test_data_label = torch.Tensor(data[1][1])

    # on crée la base de données d'apprentissage (pour torch)
    train_dataset = torch.utils.data.TensorDataset(train_data,train_data_label)
    # on crée la base de données de test (pour torch)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_data_label)
    # on crée le lecteur de la base de données d'apprentissage (pour torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # on crée le lecteur de la base de données de test (pour torch)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    # 10 fois





X = torch.Tensor(1, var.N_FEATURES + 1)  # R 1*785
Y = torch.Tensor(1, var.N_CLASSES)      # R 1*10
bias = torch.ones(1)
label = torch.Tensor(1, var.N_CLASSES)
prediction = torch.Tensor(1,var.N_CLASSES)
deltaLabel = torch.Tensor(1,var.N_CLASSES)
aux = torch.Tensor(var.N_FEATURES+1, 1)
deltaW  = torch.Tensor(var.N_FEATURES+1,var.N_CLASSES)
var.EPSILON = 0.16
y =[[] for _ in range(3)]

for j in range(100):
    W = torch.rand(var.N_FEATURES + 1, var.N_CLASSES).uniform_(-0.1, 0.1)  # R 785*10
    # var.EPSILON = float("%de%d" %(1++j/2+0.5*j+j*(7^j),-j))

    y[2].append(var.EPSILON)
    y[1].append(j)
    # [1.0, 0.7, 0.12, 0.014, 0.0015, 0.00013, 1e-05, 4e-07, 1.25e-06, 1.31e-07]
    for i in range(1):
        for i in range(var.N_IMAGES_TRAIN):
            X = torch.cat((bias,train_data[i, :]), 0)
            label =   train_data_label [i, :]
            X = X.view(1,var.N_FEATURES+1)
            prediction = torch.mm(X,W) / (var.N_FEATURES+1)
            deltaLabel = torch.add(label,prediction*-1)
            deltaLabel = deltaLabel.view(1,10)
            aux = var.EPSILON * torch.t(X)
            deltaW =  torch.mm(aux, deltaLabel)
            W = torch.add(W,deltaW)
    accurrancy= 0
    for i in range(var.N_IMAGES_TEST):
        X = torch.cat((bias, train_data[i, :]), 0)
        label = train_data_label[i, :]
        X = X.view(1,var.N_FEATURES+1)
        prediction = torch.mm(X, W) / (var.N_FEATURES+1)
     # print("predicted %f label %f" % (numpy.argmax(prediction),numpy.argmax(label)))


        if(numpy.argmax(prediction)==numpy.argmax(label)):
            accurrancy+=1
    y[0].append(accurrancy/var.N_IMAGES_TEST*100)
    print("Valeurs bien predit: %d " % (accurrancy))
    print("Valeurs mal predit:  %d " % (var.N_IMAGES_TEST))
    print("Taux de reussite:    %f" % (accurrancy/var.N_IMAGES_TEST*100))
    print("Taux d'erreur:       %f" %  (100-(accurrancy/var.N_IMAGES_TEST*100)))



# Moyenne Fitness

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.grid(True)
gridlines = ax0.get_xgridlines() + ax0.get_ygridlines()
plt.yscale('linear')
plt.xscale('linear')
print(y[2])
for line in gridlines:
    line.set_linestyle('-.')
plt.plot(y[1], y[0], 'bs',y[1], y[0])
plt.ylabel('Taux de reussite')
plt.xlabel('Iteration' )

blue_patch = mpatches.Patch(color='blue', label='Accuracy')
plt.legend(handles=[blue_patch])
plt.show()

