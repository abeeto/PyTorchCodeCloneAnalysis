# load data

import pandas as pd
data_path = 'data/international_matches.csv'
raw_data = pd.read_csv(data_path, delimiter=",")


# variables
EPOCH = 13
LEARNING_RATE = 0.01
NODES_IN_MIDDLE_LAYER = 200
NODES_IN_NARROWING = 20


teams = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands', 'England', 'Iran', 'USA', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland', 'France', 'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica', 'Germany', 'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia', 'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana', 'Uruguay', 'South Korea']
competing_teams = raw_data.loc[raw_data['home_team'].isin(teams) | raw_data['away_team'].isin(teams)]


x = competing_teams[['date', 'home_team', 'home_team_fifa_rank' ,'away_team', 'away_team_fifa_rank']]
y = competing_teams[['home_team_result']] # Win, Draw or Lose
# one-hot encode home and away teams
one_hot_home = pd.get_dummies(x['home_team'], prefix="Home")
one_hot_away = pd.get_dummies(x['away_team'], prefix="Away")


# CREATE TRAINING SET
# drop original home and away teams
x = x.drop('home_team', axis = 1)
x = x.drop('away_team', axis = 1)

# join the encoded home and away team
x = x.join(one_hot_home)
x = x.join(one_hot_away)

# helper function that turns dates into linear numbers
def convert_dates(date: str) -> int:
  date = date.split('-')
  return int(date[0]) * 365 + int(date[1]) * 31 + int(date[2])

# helper function that turns results into index outcomes
def convert_outcome(result: str) -> int:
  if result == 'Win': return 2
  if result == 'Draw': return 1
  if result == 'Lose': return 0

# convert dates to integers
for i, row in x.iterrows():
    x.at[i, 'date'] = convert_dates(row['date'])

# turn results to indexes
for i, row in y.iterrows():
    y.at[i, 'home_team_result'] = convert_outcome(row['home_team_result'])


# CREATE PYTORCH TRAINING SET
from sklearn.model_selection import train_test_split
x = x.iloc[:, :].values
y = y.iloc[:, :].values
x, X, y, Y = train_test_split(x, y, test_size=0.05)
x = x.astype(float)
X = X.astype(float)
y = y.astype(float)
Y = Y.astype(float)

import torch
x = torch.from_numpy(x)
X = torch.from_numpy(X)
y = torch.as_tensor(y, dtype=torch.float)
Y = torch.as_tensor(Y, dtype=torch.float)

from torch.utils.data import DataLoader, TensorDataset
training_data = TensorDataset(x, y)
train_loader = DataLoader(training_data , batch_size = 1, shuffle=True)
test_data = TensorDataset(X, Y)
test_loader = DataLoader(test_data , batch_size = 1, shuffle=False)


# create network
from torch import nn
class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.l1 = nn.Linear(394, NODES_IN_MIDDLE_LAYER, dtype=torch.double)
        self.l2 = nn.Linear(NODES_IN_MIDDLE_LAYER, NODES_IN_MIDDLE_LAYER, dtype=torch.double)
        self.l3 = nn.Linear(NODES_IN_MIDDLE_LAYER, NODES_IN_NARROWING, dtype=torch.double)
        self.out = nn.Linear(NODES_IN_NARROWING, 3, dtype=torch.double)
        self.a = nn.Softmax()
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.out(x)
        x = self.a(x)
        return x

model = NN()


# training
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss() # good for classification problems

for epoch in range(EPOCH):
    for data in train_loader:
        x, y = data

        optimizer.zero_grad()

        # calculate loss, backprobagation and optimize
        output = model(x)
        target = torch.Tensor([[0.0,0.0,0.0]])
        target[0][int(y.softmax(dim=0).item())] = 1.0
        loss = criterion(target, output)
        loss.backward()
        optimizer.step()    


# testing
with torch.no_grad():
    N = 0
    correct = 0
    for data in test_loader:
        x, y = data
        prediction = model(x)

        N += 1
        if int(y.item()) == prediction.argmax().item():
           correct += 1
print(f'Accuracy: {correct / N}')


# predicion

date = '2022-11-21'
fifa_rank_1 = 5
fifa_rank_2 = 100
one_hot_1 = one_hot_home.loc[one_hot_home["Home_England"] == 1].head(1)
one_hot_2 = one_hot_away.loc[one_hot_away["Away_Cameroon"] == 1].head(1)

date = convert_dates(date)

one_hot_1.index = [0]
one_hot_2.index = [0]


d = {'date': [date], 'home_team_fifa_rank': [fifa_rank_1], 'away_team_fifa_rank': [fifa_rank_2]}
df = pd.DataFrame(data=d, index=[0])
df = pd.merge(df, one_hot_1, left_index=True, right_index=True)
df = pd.merge(df, one_hot_2, left_index=True, right_index=True)


df = df.iloc[:, :].values
df = df.astype(float)
df = torch.from_numpy(df)

result = model(df).argmax(dim=1).item()
if result == 2: print(1)
if result == 1: print("X")
if result == 0: print(2)