# Projeto: 
# Estudo de implementação de uma rede neural artificial em PyTorch
# Autor: Prof. Ms. Renato Pivesso Franzin

# Objetivos:
    
"""
* Aprender os conceitos sobre redes neurais artificiais em PyTorch

* Constrir passo a passo redes neurais aplicadas em problemas de classificação 
e regressão

"""

# Descrição da Aplicação: Projeto 1: Classificação binária brest cancer
# Data: 20/01/2021
# Deep Learning de A à Z com PyTorch e Python (Aula: 6)
# Instrutor: Jones Granatyr


############################################################################## 
# -- Imports --                                                              #
############################################################################## 

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
# Versões
#print(f" PyTorch: {torch.__version__} \n ")
#!pip install torch==1.4.0

import torch.nn as nn 
# nn: neural network


############################################################################## 
# -- Seed --                                                                 #
##############################################################################

np.random.seed(123) # torna os números aleatórios previsíveis
torch.manual_seed(123) # torna os números aleatórios previsíveis

############################################################################## 
# -- Dados --                                                                #     
##############################################################################

previsores = pd.read_csv('dados\entradas_breast.csv')
classe = pd.read_csv('dados\saidas_breast.csv')


############################################################################## 
# -- Divisão Treinamento/Teste --                                            #     
##############################################################################

# 25% para teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsores, classe, test_size = 0.25)


############################################################################## 
# -- Transformação dos dados para tensores --                                #     
##############################################################################

# previsores_treinamento: DataFrame --> Array --> Tensor
previsores_treinamento = torch.tensor(np.array(previsores_treinamento), 
                                  dtype = torch.float)

# classe_treinamento: DataFrame --> Array --> Tensor
classe_treinamento = torch.tensor(np.array(classe_treinamento), 
                                  dtype = torch.float)

# dataset(previsores_treinamento, classe_treinamento)
dataset = torch.utils.data.TensorDataset(previsores_treinamento, 
                                         classe_treinamento)

train_loader = torch.utils.data.DataLoader(dataset, batch_size = 10,
                                           shuffle=True)


############################################################################## 
# -- Construção do modelo --                                                 #     
##############################################################################


# (entradas + saida) / 2 = (30 + 1) / 2 = 16

# 30 atributos
# 2 camadas ocultas de 16 neurônios

# 30 -> 16 -> 16 -> 16 -> 1

# nn: import torch.nn as nn 

classificador = nn.Sequential(
    nn.Linear(in_features=30, out_features=16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

# visualizar a estrutura do modelo
#print(classificador.parameters)


criterion = nn.BCELoss() # Cálculo do erro

optimizer = torch.optim.Adam(classificador.parameters(), lr=0.0001,
                             weight_decay=0.0001)


############################################################################## 
# -- Treinamento do modelo --                                                #     
##############################################################################

for epoch in range(1000):
  running_loss = 0.

  for data in train_loader:
    # inputs: atributos treinamento
    # labels: classes treinamento
    inputs, labels = data 
    optimizer.zero_grad() # zerar gradiente

    outputs = classificador(inputs) # classificador.forward(inputs)
    # outputs: saídas da rede neutal
   
############################################################################## 
# -- Cálculo do erro --                                                      #     
##############################################################################  
   
    loss = criterion(outputs, labels) # erro
    
    # -- Atualização dos pesos -- 
    
    loss.backward()
    optimizer.step() # peso atualizado

    running_loss += loss.item()
  print('Época %3d: perda %.5f' % (epoch+1, running_loss/len(train_loader)))
  

############################################################################## 
# -- Avaliação do modelo --                                                  #
##############################################################################

var_01 = classificador.eval()
#print(var_01)

# previsores_teste: DataFrame --> Array --> Tensor
previsores_teste = torch.tensor(np.array(previsores_teste), dtype=torch.float)

# realizar a previsão (probabilidade) em função dos atributos
previsoes = classificador(previsores_teste) 

# se previsoes_01 > 0.5, verdadeiro
# caso contrário, falso
previsoes = np.array(previsoes > 0.5) 

# -- Acurácia -- 
    
taxa_acerto = accuracy_score(classe_teste, previsoes)
print("accuracy = " ,taxa_acerto)

# -- Matriz de confusão --
matriz = confusion_matrix(classe_teste, previsoes)
print("Matriz de confusão = \n",matriz)