import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

# Entraînement réalisé soit sur la GPU ou la CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""" Choix du jeu de données normalisé et choix des hyperparamètres """
root = tk.Tk()
root.withdraw()
file_type = [("csv files", "*.csv")]
file_path = filedialog.askopenfilename(title="Choisir un jeu de données normalisé", filetypes=file_type)
root.destroy()  # Fermeture du fichier

BATCH_SIZE = int(input("Saisir la taille du batch: "))
INPUT_SIZE = int(input("Saisir le nombre de neurones de la couche d'entrée (nombre de colonnes en entrée du réseau): "))
HIDDEN_SIZE = int(input("Saisir le nombre de neurones de la couche cachée: "))
OUTPUT_SIZE = int(input("Saisir le nombre de neurones de la couche de sortie : "))
PRED_COLUMN = str(input("Saisir le nom de la colonne à prédire comme écrit dans le jeu de données: "))
LEARNING_RATE = float(input("Saisir la vitesse d'apprentissage (learning rate): "))
SPLIT = float(input("Saisir le pourcentage de données dans l'ensemble d'entraînement (ex: saisir 0.7 pour 70% des données dans l'ensemble d'entraînement): "))
PER = float(input("Saisir le pourcentage de précision entre la valeur initiale et la valeur prédite par le modèle (ex: 0.15): "))
EPOCHS = int(input("Saisir le nombre d'epochs: "))



""" Écriture de notre propre jeu de données """

class OwnDataset(Dataset):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)  # lecture du fichier csv avec des données normalisées
        self.x = torch.tensor(self.df.loc[:, self.df.columns != PRED_COLUMN].values,
                              dtype=torch.float32)  # données en entrées du réseau de neurones
        self.y = torch.tensor(self.df.loc[:, self.df.columns == PRED_COLUMN].values,
                              dtype=torch.float32)  # colonne à prédire

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

dataset = OwnDataset(file_path)


train_size = int(SPLIT * len(dataset))
test_size = len(dataset) - train_size
X_train, X_test = random_split(dataset, [train_size, test_size])

print("Nombre de données dans l'ensemble d'entraînement: ", len(X_train))
print("Nombre de données dans l'ensemble de test: ", len(X_test))

# Chargement de l'ensemble d'entraînement
train_dataloader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)



""" Création d'un ANN """

class RegressionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 1ère couche cachée
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # 2ème couche cachée
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x


# Création du modèle
model = RegressionNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
print(model)




""" Configuration d'un ANN """

erreur = nn.MSELoss()  # Régression
optimiseur = torch.optim.SGD(model.parameters(), LEARNING_RATE)




""" Apprentissage d'un ANN """


print("\n\n###################################### ENTRAÎNEMENT ######################################")
model.train()  # Activer le mode entraînement
loss_values = []
for epoch in range(0, EPOCHS):
    for i, data in enumerate(train_dataloader):  # parcours d'un batch
        # Récupérer les données dans le batch
        inputs, target = data

        # Enregistrer les tensors dans le device associé
        inputs.to(device)
        target.to(device)

        # Prédictions et calcul de l'erreur
        out = model(inputs)
        loss = erreur(out, target)

        # Mise à jour des paramètres de notre modèle
        optimiseur.zero_grad()
        loss.backward()
        optimiseur.step()

    # Enregistrement de chaque erreur
    loss_values.append((epoch + 1, round(loss.item(), 3)))

    # Affichage
    print('Epoch [{}/{}]\n Loss: {:.3f}'.format(epoch + 1, EPOCHS, loss.item()))
print("Entraînement terminé !!")

# Visualisation de la performance du modèle
epochs, losses = zip(*loss_values)
plt.plot(epochs, losses)
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.title('Erreur du modèle', fontsize=21)
plt.show()



""" Évaluation d'un ANN """

print("\n\n###################################### EVALUATION ######################################")
model.eval()
correct = 0
incorrect = 0
for data in range(len(X_test)):

    # Récupérer les données du lot
    inputs1, target1 = X_test[data]

    # Enregistrer les tensors dans le device associé
    inputs1.to(device)
    target1.to(device)

    # Predictions
    with torch.no_grad():
        out1 = model(inputs1)

    # Calcul de la différence entre le prix prédit et le prix original
    difference = np.abs(out1.item() - target1.item())

    # Maximum autorisé
    maximum = np.abs(PER * target1.item())
    if difference < maximum:
        correct += 1
    else:
        incorrect += 1
precision = (correct * 1.0) / (correct + incorrect)
print('Précision du modèle sur les {} données: {} %'.format(len(X_test), 100 * (round(precision, 3))))



""" Sauvegarder le modèle """

print("\n\n###################################### SAUVEGARDE DU MODÈLE ######################################")
torch.save(model, "Regression_ANN.pth")
print("Modèle sauvegardé !")