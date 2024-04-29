from __future__ import print_function
from __future__ import division
from torch.autograd import Variable
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
    TRAIN_BATCH_SIZE = 100


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


dtype = torch.FloatTensor
N, D_in, H, D_out = var.N_IMAGES_TRAIN,var.N_FEATURES, 300, var.N_CLASSES




model = torch.nn.Sequential(
    torch.nn.Linear(D_in, 300),
    torch.nn.ReLU(),
    torch.nn.Linear(300, 100),
    torch.nn.Linear(100, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-3
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
pl =[[] for _ in range(3)]


for t in range(100):
    for i, (data, target) in enumerate(train_loader):
        x = Variable(data)
        y =  Variable(target)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        print(t)
        print( loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pl[1].append(loss.data[0])
    pl[0].append(t)
accurrancy= 0

# x = Variable(test_data)

x = Variable(test_data)
y = Variable(test_data_label, requires_grad=False)

y_pred = model(x)

for i in range(var.N_IMAGES_TEST):
    d = y_pred[i,:]
    valuesx, indicesx = torch.max(d, 0)
    indices2 = numpy.argmax(test_data_label[i, :])
    indices1 =  indicesx.data.numpy()[0]
    print("predicted %f label %f" % (indices1,indices2  ))
    if (indices1==indices2):
        accurrancy += 1

print("Valeurs bien predit: %d " % (accurrancy))
print("Valeurs mal predit:  %d " % (var.N_IMAGES_TEST))
print("Taux de reussite:    %f" % (accurrancy/var.N_IMAGES_TEST*100))
print("Taux d'erreur:       %f" %  (100-(accurrancy/var.N_IMAGES_TEST*100)))

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.grid(True)
gridlines = ax0.get_xgridlines() + ax0.get_ygridlines()
plt.yscale('linear')
plt.xscale('linear')

print(pl)
for line in gridlines:
    line.set_linestyle('-.')
plt.plot(pl[0], pl[1], 'bs', pl[0], pl[1],markersize=2)
plt.ylabel('Erreur quadratique')
plt.xlabel('Itaration')

blue_patch = mpatches.Patch(color='blue', label='Erreur')
plt.legend(handles=[blue_patch])
plt.show()