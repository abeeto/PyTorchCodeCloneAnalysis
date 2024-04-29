import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt



""" Définition des hyperparamètres """

BATCH_SIZE = 32
INPUT_SIZE = 16
HIDDEN_SIZE = 12
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 100

# Entraînement réalisé soit sur la GPU ou le CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



""" Écriture de notre propre jeu de données """

class OwnDataset(Dataset):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)  # lecture du fichier csv avec des données normalisées
        self.x = torch.tensor(self.df.iloc[:, :-1].values,
                              dtype=torch.float32)  # données en entrées du réseau de neurones
        self.y = torch.tensor(self.df.iloc[:, -1:].values, dtype=torch.float32)  # colonne à prédire

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

dataset = OwnDataset("datasets/ParisHousingNormalise.csv")

train_size = int(0.65 * len(dataset))
test_size = len(dataset) - train_size
X_train, X_test = random_split(dataset, [train_size, test_size])

print("Nombre de données dans l'ensemble d'entraînement: ", len(X_train))
print("Nombre de données dans l'ensemble de test: ", len(X_test))

# Chargement des ensembles d'entraînement et de test
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
    (inputs1, target1) = X_test[data]

    # Enregistrer les tensors dans le device associé
    inputs1.to(device)
    target1.to(device)

    # Predictions
    with torch.no_grad():
        out1 = model(inputs1)

    # Calcul de la différence entre le prix prédit et le prix original
    difference = np.abs(out1.item() - target1.item())

    # Maximum autorisé
    maximum = np.abs(0.5 * target1.item())

    if difference < maximum:
        correct += 1
    else:
        incorrect += 1

precision = (correct * 1.0) / (correct + incorrect)
print('Précision du modèle sur les {} données: {} %'.format(len(X_test), 100 * (round(precision, 3))))



""" Sauvegarder le modèle """

torch.save(model.state_dict(), "Regression_Model_ANN.pth")
print("Modèle sauvegardé !")



""" Chargement du modèle & faire une prédiction """

model_bis = RegressionNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model_bis.load_state_dict(torch.load("Regression_Model_ANN.pth"))
model_bis.eval()
print("\n\n###################################### PREDICTION ######################################")
liste = [69990, 12, 1, 0, 5, 92000, 5, 3, 2000, 1, 0, 1000, 1597, 256, 0, 1]
norm_liste = [float(i) / sum(liste) for i in liste]
tensor = torch.tensor(norm_liste, dtype=torch.float32).to(device)
pred_label = model_bis(tensor)
pred_label = pred_label.item()
print('Prediction : {:.3f}'.format(pred_label))
print("Le prix du logement est de " + str(1000000 * round(pred_label, 3)) + " €")
