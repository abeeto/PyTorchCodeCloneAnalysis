print("INICIANDO REDE NEURAL")
print("-"*100)

#Importa as bibliotecas para realizar as redes neurais!
print("Carregando Bibliotecas")
#Torch está relacionada com o processo de REDE NEURAL
import torch

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

#Pandas é importada para dar apoio ao TORCH
import pandas as pd

#Arrays e matrizes multidimensionais são possiveis por causa do pacote NUMPY
import numpy as np

#A proxima biblioteca é para exibição grafica das respostas do sistema.
import matplotlib.pyplot as plt

print("-"*100)

print("Preparando Banco de dados")

batch_size = 100

#classe para buscar banco de dados, no formato aceito pelo torch
class BuscarBancodeDados(Dataset):
    '''Fashion MNIST Dataset'''

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file
            transform (callable): Optional transform to apply to sample
        """

        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28)  # .astype(float);
        self.Y = np.array(data.iloc[:, 0])

        del data
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            item = self.transform(item)

        return (item, label)

#Configuração do banco de dados de Treinamento e de Teste:
train_dataset = BuscarBancodeDados(csv_file='fashionmnist/fashion-mnist_train.csv')
test_dataset = BuscarBancodeDados(csv_file='fashionmnist/fashion-mnist_test.csv')

#Controladores do Banco de dados
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

print("-"*100)

print("Preparando Camadas da rede")

learning_rate = 0.001 #ACEITAÇÂO DE ERRO

num_epochs = 5 # Numero de RODADAS

def showTipos():
    labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}

    fig = plt.figure(figsize=(8,8))
    columns = 4
    rows = 5
    plt.xlabel('TIPOS')
    for i in range(1, columns*rows +1):
        img_xy = np.random.randint(len(train_dataset))
        img = train_dataset[img_xy][0][0,:,:]
        fig.add_subplot(rows, columns, i)
        plt.title((labels_map[train_dataset[img_xy][1]]).upper())
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()
showTipos()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Primeira Camada da rede
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        #Segunda Camada da rede
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(7 * 7 * 32 , 10)

    def forward(self, x):
        #TRABALHO DA REDE
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#Instancia a REDE
cnn = CNN()

#Função de Perca e Otimização
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

print("-"*100)

print("Iniciar Treinamento da Rede Neural")

losses = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.float())
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()

        outputs = cnn(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 100 == 0:
            print('Rodada : %d/%d, Iter : %d/%d,  Perda: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
print("-"*100)

print("Imprimir Tabela de Perca")
def showLosses(losses_in_epochs):
    plt.xkcd()
    plt.xlabel('Rodada #')
    plt.ylabel('Perda')
    plt.plot(losses_in_epochs)
    plt.show()

showLosses(losses[0::600])
print("-"*100)

print("Imprimindo Filtros")
def plot_kernels(tensor, num_cols=6):
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    plt.xlabel('FILTROS USADOS')
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i][0,:,:], cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

filters = cnn.modules()
model_layers = [i for i in cnn.children()]
first_layer = model_layers[0]
second_layer = model_layers[1]
first_kernels = first_layer[0].weight.data.numpy()
plot_kernels(first_kernels, 8)
second_kernels = second_layer[0].weight.data.numpy()
plot_kernels(second_kernels, 8)

plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()
print("-"*100)

print("Iniciand Teste na rede Neural")
cnn.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images.float())
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('O teste de precisão com o total de 10.000 foi de: %.4f %%' % (100 * correct / total))

labels = 'CORRETO', 'TOTAL'
sizes = [correct,total-correct]
explode = (0.09, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('TESTE DE PRECISÂO')
plt.show()

print("-"*100)
