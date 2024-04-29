import time
import numpy as np
import torch
import torch.nn as nn
import database_connection
import sys


REFERENCE_LEVEL = 1000.0


class DilatedNeuralNet(nn.Module):
    def __init__(self, dilations, hidden_sizes):
        super(DilatedNeuralNet, self).__init__()
        self.dilations = dilations
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(self.dilations)

        self.rnns = []
        parameters_list = []

        self.rnns.append([nn.RNN(input_size=1, hidden_size=hidden_sizes[0], num_layers=1, batch_first=False)
                     for _ in range(dilations[0])])
        for i in range(1, self.num_layers):
            self.rnns.append(
                [nn.RNN(input_size=hidden_sizes[i - 1], hidden_size=hidden_sizes[i], num_layers=1, batch_first=False)
                 for _ in range(dilations[i])])
            for j in range(dilations[i]):
                parameters_list += list(self.rnns[i][j].parameters())
        self.lin_layer = nn.Linear(in_features=hidden_sizes[-1], out_features=1)
        self.optimizer = torch.optim.Adam(parameters_list + list(self.lin_layer.parameters()), lr=0.01)

    def forward(self, quantities):
        neural_net_loss = torch.tensor(0.0, requires_grad = True)
        prediction = quantities[0]
        h_list = []
        for p in range(self.num_layers):
            h_list.append([torch.zeros((1, 1, hidden_sizes[p])) for _ in range(self.dilations[p])])

        for j in range(1, len(quantities)):
            neural_net_loss = neural_net_loss + (prediction - quantities[j]) ** 2
            for p in range(self.num_layers):
                current_rnn = self.rnns[p][j % dilations[p]]
                if p == 0:
                    current_quantity = torch.tensor(quantities[j])
                    current_quantity = current_quantity.view(1, 1, 1)
                    prediction, h_list[0][j % dilations[0]] = current_rnn(current_quantity, h_list[0][j % dilations[0]])
                elif p > 0:
                    prediction, h_list[p][j % dilations[p]] = current_rnn(h_list[p-1][j % dilations[p-1]],
                                                                          h_list[p][j % dilations[p]])
            prediction = self.lin_layer(prediction)

        return prediction, neural_net_loss

    def save_to_file(self, path_to_file):
        # forming dict to save
        model_dict = {}
        for i in range(len(self.rnns)):
            for j in range(self.dilations[i]):
                model_dict[(i, j)] = self.rnns[i][j].state_dict()
        model_dict['lin_layer'] = self.lin_layer.state_dict()
        torch.save(model_dict, path_to_file)

    def get_model_from_file(self, path_to_file):
        checkpoint = torch.load(path_to_file)
        for i in range(len(self.rnns)):
            for j in range(self.dilations[i]):
                self.rnns[i][j].load_state_dict(checkpoint[(i, j)])
                self.lin_layer.load_state_dict(checkpoint['lin_layer'])
        # model.load_state_dict(torch.load(path))


def get_time_series_list(articule):
    conn = database_connection.DatabaseConnection()
    filials_set = conn.get_filials_for_articule(articule, start_date, end_date, min_days=40)
    silpo_fora_trash_filials_set = conn.get_silpo_fora_trash_filials_set()
    filials_set = filials_set.intersection(silpo_fora_trash_filials_set)

    print("Number of filials:", len(filials_set))

    list_of_time_series = []

    filials_list = []

    epsilon = 0.1
    for filial in filials_set:
        df = conn.get_all_sales_by_articule_and_filial_with_residues(articule, filial, start_date, end_date)
        normed_quantities = (df['quantity'].values + epsilon) / REFERENCE_LEVEL
        is_promo_day = df['is_promo_day'].values
        residues = df['residue'].values
        normed_quantities = normed_quantities[(np.logical_not(is_promo_day)) & (residues>0)]
        if len(normed_quantities) > 20:
            list_of_time_series.append(normed_quantities)
            filials_list.append(filial)
    return filials_list, list_of_time_series


start_date = "'2018-04-01'"
end_date = "'2018-08-31'"

# start_date = "'2019-04-01'"
# end_date = "'2019-08-31'"

articule = 32485  # 361771 519429 97143 32547 32485 117

filials_list, list_of_time_series = get_time_series_list(articule)
start_time = time.time()

dilations = [1, 2, 4]
hidden_sizes = [3, 2, 3]

rnn = DilatedNeuralNet(dilations, hidden_sizes)

# optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

num_epochs = 10
for i in range(num_epochs):
    for k in range(len(filials_list)):
        quantities = list_of_time_series[k]
        _, neural_net_loss = rnn.forward(quantities)
        rnn.optimizer.zero_grad()
        neural_net_loss.backward()
        rnn.optimizer.step()

# see that in this way we do not get all the parameters we need
print("SD:", rnn.state_dict())

# this does not help much:
# torch.save(rnn.state_dict(), path)

path = r"D:\PycharmProjects\ExperimentsWithPyTorch\rnn_states.pt"

rnn.save_to_file(path)

for k in range(8):
    filial = filials_list[k]
    quantities = list_of_time_series[k]
    prediction, _ = rnn.forward(quantities)
    print(f"Prediction for filial {filial}: {prediction * REFERENCE_LEVEL}")

model = DilatedNeuralNet(dilations, hidden_sizes)
model.get_model_from_file(path)

print()
for k in range(8):
    filial = filials_list[k]
    quantities = list_of_time_series[k]
    prediction, _ = model.forward(quantities)
    print(f"New prediction for filial {filial}: {prediction * REFERENCE_LEVEL}")

print("Finished!")
seconds = time.time() - start_time
hours = seconds / 3600
minutes = seconds / 60
print(f"Neural net learning took {hours:.2f} hours")
print(f"Neural net learning took {minutes:.2f} minutes")
print(f"Neural net learning took {seconds:.2f} seconds")
