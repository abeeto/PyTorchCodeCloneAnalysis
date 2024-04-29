import pandas as pd

# all_data = pd.read_csv("Data_Entry_2017.csv")
# train = pd.read_csv("train_val_list.txt")
# train.columns = ["Image Index"]
# test = pd.read_csv("test_list.txt")
# test.columns = ["Image Index"]

## Save train and test metadata sets
# all_data.merge(train, on="Image Index").to_csv("train_metadata.csv", index=False)
# all_data.merge(test, on="Image Index").to_csv("test_metadata.csv", index=False)

# Create validation set
train_data = pd.read_csv("train_metadata.csv")
labels_dict = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3,
               "Mass": 4, "Nodule": 5, "Pneumonia": 6, "Pneumothorax": 7}

n_samples = 200

data = train_data.copy()
for i, label in enumerate(labels_dict.keys()):

    subset = data[data['Finding Labels'].str.contains(label)].sample(n=n_samples, random_state=42)

    if i == 0:
        val_data = subset
    else:
        val_data = val_data.append(subset)

    data = data[~data.index.isin(subset.index)]

val_data.to_csv("validation_metadata.csv", index=False)
