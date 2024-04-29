import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt


# Entraînement réalisé soit sur la GPU ou la CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




""" Définition des hyperparamètres """

BATCH_SIZE = 32
INPUT_SIZE = 20
HIDDEN_SIZE = 15
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 100




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

dataset = OwnDataset("datasets/WaterQualityNormalise.csv")

# 80% des données pour l'ensemble d'entraînement & 20% des données pour l'ensemble de test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
X_train, X_test = random_split(dataset, [train_size, test_size])

print("Nombre de données dans l'ensemble d'entraînement: ", len(X_train))
print("Nombre de données dans l'ensemble de test: ", len(X_test))

# Chargement des ensembles d'entraînement et de test
train_dataloader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=True)



""" Création d'un ANN """

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Unique couche cachée
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x


# Création du modèle
model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
print(model)



""" Configuration d'un ANN """

erreur = nn.BCELoss()  # Classification binaire
optimiseur = torch.optim.Adam(model.parameters(), LEARNING_RATE)



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
        outputs = model(inputs)
        loss = erreur(outputs, target)

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
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_dataloader:

        # Récupérer les données du lot
        inputs1, target1 = data

        # Enregistrer les tensors dans le device associé
        inputs1.to(device)
        target1.to(device)
        outputs1 = model(inputs1.float())

        # Calcul de la précision
        for idx, i in enumerate(outputs1):
            if torch.argmax(i) == target1[idx]:
                correct += 1
            total += 1

    print('Précision du modèle sur les {} données: {} %'.format(len(X_test), 100 * (round(correct / total, 3))))



""" Sauvegarder le modèle """

print("\n\n###################################### SAUVEGARDE DU MODÈLE ######################################")
torch.save(model.state_dict(), "Binary_Classification_Model_ANN.pth")
print("Modèle sauvegardé !")



""" Chargement du modèle & faire une prédiction """

model_bis = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model_bis.load_state_dict(torch.load("Binary_Classification_Model_ANN.pth"))
model_bis.eval()
print("\n\n###################################### PREDICTION ######################################")
# Valeurs aléatoires mises par moi-même pour observer si le modèle prédira bien que l'eau est saine
liste = [1.32, 16.25, 0.2, 3, 0.001, 3, 0.05, 1.22, 1.35, 0, 0, 0.012, 8, 0.75, 0.001, 25, 3.5, 0.03, 0.087, 0.3]
norm_liste = [float(i) / sum(liste) for i in liste]
tensor = torch.tensor(norm_liste, dtype=torch.float32).to(device)
pred_label = model_bis(tensor)
pred_label = pred_label.item()
print('Prediction : {:.3f}'.format(pred_label))
if pred_label >= 0.5:
    print("L'eau est saine à " + str(100 * round(pred_label, 3)) + " %")
else:
    print("L'eau n'est pas saine du tout")