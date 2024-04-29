import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

# Entraînement réalisé soit sur la GPU ou la CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Choix du fichier des labels & choix du dossier des images """

# Choix du fichier .csv contenant les labels normalisés
root = tk.Tk()
root.withdraw()
files_types = [("csv file", "*.csv")]
file_path = filedialog.askopenfilename(title="Choisir le fichier csv contenant les labels", filetypes=files_types)
root.destroy()  # Fermeture de la fenêtre

# Choix du dossier des images
root1 = tk.Tk()
root1.withdraw()
folder_path = filedialog.askdirectory(title="Choisir le dossier des images")
root1.destroy()  # Fermeture de la fenêtre

""" Choix des hyperparamètres """

SPLIT = float(input(
    "Saisir le pourcentage de données dans l'ensemble d'entraînement (ex: saisir 0.7 pour 70% des données dans l'ensemble d'entraînement): "))
BATCH_SIZE = int(input("Saisir la taille du batch: "))
LABELS_COLUMN = int(input("Saisir l'index de la colonne des labels: "))
ID_COLUMN = int(input("Saisir l'index de la colonne des identifiants (noms des images): "))
IMAGE_SIZE = int(input("Saisir la taille de l'image: "))
IMAGE_CHANNELS = int(input("Saisir le nombre de canaux de votre image (3 pour RGB (couleurs) ou 1 (grayscale)): "))
IMAGE_SIZE2 = int(input("Saisir la taille de l'image en sortie de la seconde couche de convolution: "))
IMAGE_SIZE3 = int(input("Saisir la taille de l'image en sortie de la troisième couche de convolution: "))
IMAGE_SIZE4 = int(input("Saisir la taille de l'image en sortie de la quatrième couche de convolution: "))
KERNEL_SIZE = int(input("Saisir la taille du kernel (filtre): "))
STRIDE_SIZE = int(input("Définir le pas du kernel lorsqu'il se déplace sur l'image (ex: un pas de 1 signifie que le kernel se déplace pixel par pixel): "))
INPUT_SIZE = int(input("Saisir le nombre de neurones en entrée de la couche fully connected (si vous ne connaissez pas la formule, mettez une valeur aléatoire une erreur s'affichera 'mat1 and mat2 shapes cannot be multiplied' pour voir la réélle valeur de l'input_size qui correspond la colonne (format: nrow x ncol) de la mat1): "))
OUTPUT_SIZE = int(input("Saisir le nombre de neurones en sortie de la couche fully connected: "))
NUM_CLASSES = int(input("Saisir le nombre de classes/labels: "))
LEARNING_RATE = float(input("Saisir la vitesse d'apprentissage (learning rate): "))
EPOCHS = int(input("Saisir le nombre d'epochs: "))

""" Écriture de notre propre jeu de données """

class ImageDataset(Dataset):
    def __init__(self, label_file, root_dir, transform):
        self.df = pd.read_csv(label_file)  # Ouverture du fichier contenant les labels normalisés
        self.folder_dir = root_dir  # Chemin vers le dossier contenant les images à classer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, ID_COLUMN]  # colonne des identifiants des images
        img_name = os.path.join(self.folder_dir,
                                img_id)  # Récupération des images
        imgs_set = PIL.Image.open(img_name).convert('RGB')  # Ouverture des images
        label = torch.tensor(self.df.iloc[idx, LABELS_COLUMN],
                             dtype=torch.float32)  # colonne contenant les labels

        # Transformation des images
        if self.transform is not None:
            imgs_set = self.transform(imgs_set)

        return imgs_set, label

# Redimensionner l'image, convertir les PIL Images en tensor, convertir le type des images tensor en float32, normalisation des tensors
all_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

image_dataset = ImageDataset(label_file=file_path, root_dir=folder_path, transform=all_transforms)

# Division du jeu de données en ensemble de test et d'entraînement
train_size = int(SPLIT * len(image_dataset))
test_size = len(image_dataset) - train_size
X_train, X_test = random_split(image_dataset, [train_size, test_size])

print("\nNombre de données dans l'ensemble d'entraînement: ", len(X_train))
print("Nombre de données dans l'ensemble de test: ", len(X_test))

# Chargement de l'ensemble d'entraînement et l'ensemble de test
train_dataloader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=True)

""" Création d'un CNN """

class ConvNeuralNet(nn.Module):
    def __init__(self, imgs_channels, taille_image, size_kernel, taille_image2, stride_size, taille_image3, taille_image4, input_size, output_size, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=imgs_channels, out_channels=taille_image, kernel_size=size_kernel)
        self.conv_layer2 = nn.Conv2d(in_channels=taille_image, out_channels=taille_image2, kernel_size=size_kernel)
        self.max_pool1 = nn.MaxPool2d(kernel_size=size_kernel, stride=stride_size)

        self.conv_layer3 = nn.Conv2d(in_channels=taille_image2, out_channels=taille_image3, kernel_size=size_kernel)
        self.conv_layer4 = nn.Conv2d(in_channels=taille_image3, out_channels=taille_image4, kernel_size=size_kernel)
        self.max_pool2 = nn.MaxPool2d(kernel_size=size_kernel, stride=stride_size)

        self.fc1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, num_classes)


    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# Création du modèle
model = ConvNeuralNet(imgs_channels=IMAGE_CHANNELS, taille_image=IMAGE_SIZE, size_kernel=KERNEL_SIZE, taille_image2=IMAGE_SIZE2, stride_size=STRIDE_SIZE, taille_image3=IMAGE_SIZE3, taille_image4=IMAGE_SIZE4, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, num_classes=NUM_CLASSES)
print(model)

""" Configuration d'un CNN """

criterion = nn.CrossEntropyLoss()  # Classification multi-classes
optimiseur = torch.optim.SGD(model.parameters(), LEARNING_RATE)

""" Apprentissage d'un CNN """

print("\n\n###################################### ENTRAÎNEMENT ######################################")
model.train()  # Activer le mode entraînement
loss_values = []
for epoch in range(0, EPOCHS):
    for i, (images, labels) in enumerate(train_dataloader):
        # Enregistrer les tensors dans le device associé
        images.to(device)
        labels.to(device)

        # Prédictions et calcul de l'erreur
        output = model(images)
        loss = criterion(output, labels.long())

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

""" Évaluation d'un CNN """

print("\n\n###################################### EVALUATION ######################################")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images.to(device)
        labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Précision du modèle sur les {} images: {} %'.format(len(X_test), 100 * (round(correct / total, 3))))

""" Sauvegarder le modèle """

print("\n\n###################################### SAUVEGARDE DU MODÈLE ######################################")
torch.save(model, "Classification_CNN.pth")
print("Modèle sauvegardé !")
