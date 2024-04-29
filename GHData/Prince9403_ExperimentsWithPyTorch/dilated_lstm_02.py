import sys
import time
import numpy as np
import torch
import torch.nn as nn
import database_connection


REFERENCE_LEVEL = 1000.0


class DilatedLSTM:
    def __init__(self, dilations, hidden_sizes):
        self.dilations = dilations
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(self.dilations)

        self.rnns = []
        parameters_list = []

        self.rnns.append([nn.LSTM(input_size=1, hidden_size=hidden_sizes[0], num_layers=1, batch_first=False)
                     for _ in range(dilations[0])])
        for i in range(1, self.num_layers):
            self.rnns.append(
                [nn.LSTM(input_size=hidden_sizes[i - 1], hidden_size=hidden_sizes[i], num_layers=1, batch_first=False)
                 for _ in range(dilations[i])])
            for j in range(dilations[i]):
                parameters_list += list(self.rnns[i][j].parameters())
        self.lin_layer = nn.Linear(in_features=hidden_sizes[-1], out_features=1)
        self.optimizer = torch.optim.Adam(parameters_list + list(self.lin_layer.parameters()), lr=0.01)

    def forward(self, quantities):
        neural_net_loss = torch.tensor(0.0, requires_grad = True)
        prediction = quantities[0]
        h_list = []
        c_list = []
        for p in range(num_layers):
            h_list.append([torch.zeros((1, 1, hidden_sizes[p])) for _ in range(dilations[p])])
            c_list.append([torch.zeros((1, 1, hidden_sizes[p])) for _ in range(dilations[p])])

        for j in range(1, len(quantities)):
            neural_net_loss = neural_net_loss + (prediction - quantities[j]) ** 2
            for p in range(num_layers):
                current_rnn = self.rnns[p][j % dilations[p]]
                if p == 0:
                    current_quantity = torch.tensor(quantities[j])
                    current_quantity = current_quantity.view(1, 1, 1)
                    prediction, (h_list[0][j % dilations[0]], c_list[0][j % dilations[0]]) = \
                        current_rnn(current_quantity, (h_list[0][j % dilations[0]], c_list[0][j % dilations[0]]))
                elif p > 0:
                    prediction, (h_list[p][j % dilations[p]], c_list[p][j % dilations[p]]) = \
                        current_rnn(h_list[p-1][j % dilations[p-1]],
                                    (h_list[p][j % dilations[p]], c_list[p][j % dilations[p]]))
            prediction = self.lin_layer(prediction)
        return prediction, neural_net_loss

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

rnn = DilatedLSTM(dilations, hidden_sizes)

num_layers = len(dilations)

num_epochs = 10
for i in range(num_epochs):
    for k in range(len(filials_list)):
        filial = filials_list[k]
        quantities = list_of_time_series[k]
        _, neural_net_loss = rnn.forward(quantities)
        rnn.optimizer.zero_grad()
        neural_net_loss.backward()
        rnn.optimizer.step()

print("Finished!")
seconds = time.time() - start_time
hours = seconds / 3600
minutes = seconds / 60
print(f"Neural net learning took {hours:.2f} hours")
print(f"Neural net learning took {minutes:.2f} minutes")
print(f"Neural net learning took {seconds:.2f} seconds")
