from numpy import mean, corrcoef
import pandas as pd
import random
import json
import matplotlib as plt
import plotly.graph_objs as go
import plotly.plotly as py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# determines the winning team based on player scores
def winner(players):
    return any([p['score'] == 10 for p in players])


# confirms a match as valid
def validate(players):
    for player in players:
        if player is None:                  # player not present
            return False
        if player['afterMean'] is None:     # not rated
            return False
        if player['faction'] > 4:           # modded faction
            return False
    if not any([p['score'] == 10 for p in players]):  # corrupt replay
        return False
    return True


# class for reshaping within sequential net
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class SetonsDataset(Dataset):
    def __init__(self, filename):
        print('reading...')
        self.data = []
        file = open(filename, 'r').read()
        df = pd.DataFrame.from_dict(json.loads(file))
        games = list(df.groupby(
            df.relationships.apply(lambda x: x['game']['data']['id']),
            sort=False))

        print('ingesting...')
        for game in games:
            gid, frame = game
            players = [None] * 8
            for a, i, r, t in frame.values:
                position = a['startSpot'] - 1
                players[position] = a
            if validate(players):
                # 2x4x8 -> mean-dev xx faction xx position
                x = [[[0, 0, 0, 0, 0, 0, 0, 0],  # mean
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]],
                     [[0, 0, 0, 0, 0, 0, 0, 0],  # deviation
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]]
                y = winner(players[0::2])
                for i in range(8):  # fill in x
                    f = players[i]['faction'] - 1
                    m = players[i]['afterMean']
                    d = players[i]['afterDeviation']
                    x[0][f][i] = m
                    x[1][f][i] = d
                self.data.append((torch.FloatTensor(x), y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_printoptions(threshold=5000, precision=3, linewidth=200)

    net = nn.Sequential(
        nn.BatchNorm2d(2),  # Bx2x4x8
        View(-1, 64),       # 2x4x8 -> 1x1x64
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        View(-1),           # 1x1 int array -> int
        nn.Sigmoid()        # output layer
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    data = SetonsDataset('setons.json')
    len_data = len(data)
    epochs = 50

    print(len_data, 'cases')
    print('cuda: ', use_cuda)

    graph_percentage = []
    graph_correlation = []
    for epoch in range(epochs):
        training_gen = DataLoader(data[1000:], batch_size=50, shuffle=True, num_workers=2)
        validation_gen = DataLoader(data[:1000], num_workers=2)

        # training
        for batch, batch_labels in training_gen:
            # batch, batch_labels = batch.to(device), batch_labels.to(device)
            y_pred = net(batch)
            loss = loss_fn(y_pred, batch_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        predictions = []
        labels = []
        coin_flip = []
        for match, label in validation_gen:
            # match, label = match.to(device), label.to(device)
            predictions.append(float(net(match)))
            labels.append(bool(label))
            coin_flip.append((predictions[-1] > .5) == bool(label))
        graph_percentage.append(mean(coin_flip))
        graph_correlation.append(corrcoef(predictions, labels)[0][1])
        print(epoch, graph_percentage[-1], graph_correlation[-1])

    ''' Analysis '''

    netvtime = [
        go.Scatter(
            name='percentage',
            x=list(range(len(graph_percentage))),
            y=graph_percentage,
            mode='lines'
        ),
        go.Scatter(
            name='correlation',
            x=list(range(len(graph_correlation))),
            y=graph_correlation,
            mode='lines'
        )
    ]
    py.plot(netvtime, filename='netvtime')
