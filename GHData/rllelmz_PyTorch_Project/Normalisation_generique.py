import tkinter as tk
from tkinter import filedialog
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype
from sklearn import preprocessing

# Choix du fichier csv à lire
root = tk.Tk()
root.withdraw()
file_type = [("csv files", "*.csv")]
file_path = filedialog.askopenfilename(title="Choisir un fichier", filetypes=file_type)

# Lire le fichier csv
df = pd.read_csv(file_path)

# Preprocessing
df1 = df.dropna(axis=0, how='any')
print(df1.head(10))

# Fonction normalisation
def min_max_scaling(column):
    return round((column - column.min()) / (column.max() - column.min()), 3)

# Encoder les variables catégorielles si elles existent et normaliser les données
for col in df1:
    if is_string_dtype(df1[col]):
        df1[col] = preprocessing.LabelEncoder().fit_transform(df1[col])
        df1[col] = min_max_scaling(df1[col])
    elif is_bool_dtype(df1[col]):
        df1[col] = preprocessing.LabelEncoder().fit_transform(df1[col])
        df1[col] = min_max_scaling(df1[col])
    elif is_numeric_dtype(df1[col]):
        df1[col] = min_max_scaling(df1[col])
    else:
        df1.drop(col, axis=1, inplace=True)


# Enregistrement de la dataframe normalisé dans un fichier csv
try:
    with filedialog.asksaveasfile(mode="w", defaultextension=".csv") as file:
        df1.to_csv(file.name, index=False)
        print("Fichier crée !!")
except AttributeError:
    print("Fichier non sauvegardé !!")

root.destroy()  # Fermeture de la fenêtre

