import numpy as np
from dataset import ChatDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NeuralNet, device

# Se carga el dataset y se define el tamaño de las muestras
dataset = ChatDataset()
batch_size = 8
train_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0) 

# Hiper parametros
hidden_size = 10
output_size = len(dataset.tags)
input_size = len(dataset.x_data[0])
learning_rate = 0.001
num_epoch = 1200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Se inicializa el modelo
model = NeuralNet(input_size, hidden_size,output_size).to(device)

# Definicion de criterios de Perdida y optimización
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Iniciando entrenamiento...")
for epoch in range(num_epoch):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # propagación
        output = model(words)
        loss = criterion(output, labels)
        
        # optimización y función de perdida
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1}/{num_epoch} \
                loss={loss.item():.4f}")

print("Entrenamiento finalizado.")

# Modelo listo para guardar
data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":dataset.all_words,
    "tags":dataset.tags,
    "final_loss":loss.item()
}

FILE = "brain.pth"

torch.save(data, FILE)

print("Modelo guardado.")
