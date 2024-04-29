###import required libraries

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###Data load section

df = pd.read_csv('NYCTaxiFares.csv')
print(df.head())

print(df['fare_amount'].describe())


###Calculate the distance traveled

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers

    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)  # in kilometers
    return d


df['dist_km'] = haversine_distance(df,'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
print(df.head())

###Add a datetime column and derive usefull statistics

df['EDTdate'] = pd.to_datetime(df['pickup_datetime'].str[:19]) - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour']<12,'am','pm')
df['Weekday'] = df['EDTdate'].dt.strftime("%a")
print(df.head())

print(df['EDTdate'].min())
print(df['EDTdate'].max())

##seperate categorical and continuos columns

print(df.columns)

cat_cols = ['Hour', 'AMorPM', 'Weekday']
cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount']  # this column contains the labels
print(cat_cols)


###Convert our three categorical columns to category dtypes.
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
print(df.dtypes)

print(df['Hour'].head())
print(df['AMorPM'].head())
print(df['AMorPM'].cat.categories)
print(df['AMorPM'].head().cat.codes)
print(df['Weekday'].cat.categories)
print(df['Weekday'].head().cat.codes)


##Now we want to combine the three
# categorical columns into one input
# array using numpy.stack We don't want the Series index,
# just the values.

hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkdy], 1)

print(cats[:5])

###convert numpy arrays to tensor

cats = torch.tensor(cats, dtype=torch.int64)
# this syntax is ok, since the source data is an array, not an existing tensor

print(cats[:5])

###convert continuosu variables to a sensor
conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)
print(conts[:5])


### Convert labels to a tensor
y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1,1)

print(y[:5])

print(cats.shape)
print(conts.shape)
print(y.shape)


###set an embedding size

#The rule of thumb for determining the embedding size is to
#divide the number of unique entries in each column by 2, but not to exceed 50.

# This will set embedding sizes for Hours, AMvsPM and Weekdays
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)



###define a tabular Model
#This somewhat follows the fast.ai library The goal is to define a model
# based on the number of continuous columns (given by conts.shape[1]) plus
# the number of categorical columns and their embeddings (given by len(emb_szs)
# and emb_szs respectively). The output would either be a regression (a single
# float value), or a classification (a group of bins and their softmax values).
# For this exercise our output will be a single regression value. Note that
# we'll assume our data contains both categorical and continuous data. You
# can add boolean parameters to your own model class to handle a variety of
# datasets

class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layerlist = []
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


torch.manual_seed(33)
model = TabularModel(emb_szs, conts.shape[1], 1, [200,100], p=0.4)
print(model)

###define loss function and optimizer

#PyTorch does not offer a built-in RMSE Loss
# function, and it would be nice to see this in place of MSE.
#For this reason, we'll simply apply the torch.sqrt() function to the output
# of MSELoss during training.

criterion = nn.MSELoss()  # we'll convert this to RMSE later
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


###perform train , test splits

#At this point our batch size is the entire dataset of 120,000 records. This will take a long time to train, so you might consider reducing t
# his. We'll use 60,000. Recall that our tensors are already randomly shuffled.

batch_size = 60000
test_size = int(batch_size * .2)

cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

print(len(cat_train))
print(len(cat_test))


##Train the model

import time

start_time = time.time()

epochs = 50
losses = []

for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train))  # RMSE
    losses.append(loss)

    # a neat trick to save screen space:
    if i % 25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}')  # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed


###plot the loss function

plt.plot(range(epochs), losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch');
plt.show()


##validate the model

#Here we want to run the entire test set through the model, and compare it to the known labels.
#For this step we don't want to update weights and biases, so we set torch.no_grad()


with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))
print(f'RMSE: {loss:.8f}')


#Now let's look at the first 50 predicted values:

print(f'{"PREDICTED":>12} {"ACTUAL":>8} {"DIFF":>8}')
for i in range(50):
    diff = np.abs(y_val[i].item()-y_test[i].item())
    print(f'{i+1:2}. {y_val[i].item():8.4f} {y_test[i].item():8.4f} {diff:8.4f}')


#So while many predictions were off by a few cents, some
# were off by $19.00. Feel free to change the batch size, test size,
# and number of epochs to obtain a better model.
