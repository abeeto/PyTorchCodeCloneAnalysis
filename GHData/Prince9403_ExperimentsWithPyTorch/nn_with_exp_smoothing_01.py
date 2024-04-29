import time
import numpy as np
import torch
import torch.nn as nn
import database_connection
import sys


def get_time_series_list(articule):
    reference_level = 1000.0

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
        normed_quantities = (df['quantity'].values + epsilon) / reference_level
        is_promo_day = df['is_promo_day'].values
        residues = df['residue'].values
        normed_quantities = normed_quantities[(np.logical_not(is_promo_day)) & (residues>0)]
        if len(normed_quantities) > 20:
            list_of_time_series.append(normed_quantities)
            filials_list.append(filial)
    return filials_list, list_of_time_series


def training_step(time_series, beta, hidden_size, num_layers, rnn, lin_layer,
                  beta_optimizer, rnn_optimizer):
    neural_net_loss = torch.tensor(0.0)
    h = torch.zeros(hidden_size * num_layers)
    h = h.view((num_layers, 1, hidden_size))
    c = torch.zeros(hidden_size * num_layers)
    c = c.view((num_layers, 1, hidden_size))

    rnn_input = torch.zeros(len(time_series) - 1)
    expected_outputs = torch.zeros(len(time_series) - 1)
    S = torch.tensor(quantities[0])  # requires_grad=True
    alpha = torch.sigmoid(beta)

    for j in range(len(time_series) - 1):
        rnn_input[j] = torch.log(time_series[j] / S)
        S = (1 - alpha) * S + alpha * time_series[j]  # current_quantity
        expected_outputs[j] = torch.log(time_series[j + 1] / S)
    # print(f"Input shape: {rnn_input.shape}")
    rnn_input = rnn_input.view((len(time_series) - 1, 1, 1))
    rnn_output, (h, c) = rnn(rnn_input, (h, c))
    rnn_output = lin_layer(rnn_output)
    rnn_output = rnn_output.view(len(time_series) - 1)
    rnn_loss = (torch.abs(rnn_output - expected_outputs)).mean()
    # print(f"rnn_loss={rnn_loss} Does rnn loss require grad? {rnn_loss.requires_grad}")
    assert not np.isnan(neural_net_loss.detach()), \
        f"Sorry, neural_net_loss is nan"
    beta_optimizer.zero_grad()
    rnn_optimizer.zero_grad()
    rnn_loss.backward()
    beta_optimizer.step()
    rnn_optimizer.step()
    assert not np.isnan(beta.detach()), f"Sorry, beta is nan"


start_date = "'2018-04-01'"
end_date = "'2018-08-31'"

# start_date = "'2019-04-01'"
# end_date = "'2019-08-31'"

articule = 32485  # 361771 519429 97143 32547 32485 117

filials_list, list_of_time_series = get_time_series_list(articule)
start_time = time.time()

# now we want to find optimal alpha
beta = torch.tensor(0.5, requires_grad=True)
alpha = torch.sigmoid(beta)
beta_optimizer = torch.optim.Adam([beta], lr=0.01)

hidden_size = 3
num_layers = 4
# rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
lin_layer = nn.Linear(in_features=hidden_size, out_features=1)
rnn_optimizer = torch.optim.Adam(list(rnn.parameters()) + list(lin_layer.parameters()), lr=0.01)

num_epochs = 100
for i in range(num_epochs):
    for k in range(len(filials_list)):
        filial = filials_list[k]
        quantities = list_of_time_series[k]
        training_step(quantities, beta, hidden_size, num_layers, rnn, lin_layer,
                      beta_optimizer, rnn_optimizer)
        assert not np.isnan(beta.detach()), f"Sorry, beta is nan. Articule {articule}, filial {filial}, i={i}, k={k}"

alpha = torch.sigmoid(beta)
print(f"Articule: {articule} alpha: {alpha} beta: {beta}")
assert not np.isnan(beta.detach()), "Sorry, beta is nan"
assert not np.isnan(alpha.detach()), "Sorry, alpha is nan"

print("Finished!")
seconds = time.time() - start_time
hours = seconds / 3600
minutes = seconds / 60
print(f"Neural net learning took {hours:.2f} hours")
print(f"Neural net learning took {minutes:.2f} minutes")
print(f"Neural net learning took {seconds:.2f} seconds")
