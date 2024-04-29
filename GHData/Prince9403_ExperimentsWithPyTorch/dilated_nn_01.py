import time
import numpy as np
import torch
import torch.nn as nn
import database_connection
import sys


REFERENCE_LEVEL = 1000.0

class DilatedNeuralNet(nn.Module):
    def __init__(self, hidden_size_0, hidden_size_1):
        super(DilatedNeuralNet, self).__init__()
        self.rnn_0_0 = nn.RNN(input_size=1, hidden_size=hidden_size_0, num_layers=1, batch_first=False)
        self.rnn_0_1 = nn.RNN(input_size=1, hidden_size=hidden_size_0, num_layers=1, batch_first=False)
        self.rnn_1_0 = nn.RNN(input_size=hidden_size_0, hidden_size=hidden_size_1, num_layers=1, batch_first=False)
        self.lin_layer = nn.Linear(in_features=hidden_size_1, out_features=1)

    def forward(self, quantities):
        loss = torch.tensor(0.0, requires_grad=True)
        prediction = quantities[0]
        h_0_0 = torch.zeros((1, 1, hidden_size_0))
        h_0_1 = torch.zeros((1, 1, hidden_size_0))
        h_1_0 = torch.zeros((1, 1, hidden_size_1))

        for j in range(1, len(quantities)):
            loss = loss + (prediction - quantities[j]) ** 2
            current_quantity = torch.tensor(quantities[j])
            current_quantity = current_quantity.view(1, 1, 1)
            if j % 2 == 0:
                _, h_0_0 = self.rnn_0_0(current_quantity, h_0_0)
                prediction, h_1_0 = self.rnn_1_0(h_0_0, h_1_0)
            elif j % 2 == 1:
                _, h_0_1 = self.rnn_0_1(current_quantity, h_0_1)
                prediction, h_1_0 = self.rnn_1_0(h_0_1, h_1_0)
            prediction = self.lin_layer(prediction)
        return prediction, loss


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

hidden_size_0 = 3
hidden_size_1 = 3
# num_layers = 4

rnn = DilatedNeuralNet(hidden_size_0=hidden_size_0, hidden_size_1=hidden_size_1)
rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

num_epochs = 10
for i in range(num_epochs):
    for k in range(len(filials_list)):
        quantities = list_of_time_series[k]
        _, neural_net_loss = rnn.forward(quantities)
        rnn_optimizer.zero_grad()
        neural_net_loss.backward()
        rnn_optimizer.step()

for k in range(4):
    filial = filials_list[k]
    quantities = list_of_time_series[k]
    prediction, _ = rnn.forward(quantities)
    print(f"Prediction for filial {filial}: {prediction * REFERENCE_LEVEL}")

path = r"D:\PycharmProjects\ExperimentsWithPyTorch\rnn_states_01.pt"
torch.save(rnn.state_dict(), path)

model = DilatedNeuralNet(hidden_size_0=hidden_size_0, hidden_size_1=hidden_size_1)
model.load_state_dict(torch.load(path))

print()
for k in range(4):
    filial = filials_list[k]
    quantities = list_of_time_series[k]
    prediction, _ = model.forward(quantities)
    print(f"Prediction of saved model for filial {filial}: {prediction * REFERENCE_LEVEL}")


print("Finished!")
seconds = time.time() - start_time
hours = seconds / 3600
minutes = seconds / 60
print(f"Neural net learning took {hours:.2f} hours")
print(f"Neural net learning took {minutes:.2f} minutes")
print(f"Neural net learning took {seconds:.2f} seconds")
