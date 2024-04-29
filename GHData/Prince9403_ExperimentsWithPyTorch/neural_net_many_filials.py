import time
import sys
import torch
import torch.nn as nn
import database_connection


start_time = time.time()

start_date = "'2019-04-01'"
end_date = "'2019-08-31'"

articule = 32485  # 361771 519429 97143 32547 32485 117

reference_level = 1.0  # 1000.0

conn = database_connection.DatabaseConnection()
filials_set = conn.get_filials_for_articule(articule, start_date, end_date, min_days=40)
silpo_fora_trash_filials_set = conn.get_silpo_fora_trash_filials_set()
filials_set = filials_set.intersection(silpo_fora_trash_filials_set)

print("Number of filials:", len(filials_set))

list_of_time_series = []

filials_list = list(filials_set)

for filial in filials_list:
    df = conn.get_all_sales_by_articule_and_filial_with_residues(articule, filial, start_date, end_date)
    list_of_time_series.append(df['quantity'].values/reference_level)

rnn = nn.RNN(input_size=1, hidden_size=2, num_layers=1, batch_first=False)
lin_layer = nn.Linear(in_features=2, out_features=1)
optimizer = torch.optim.Adam(list(rnn.parameters()) + list(lin_layer.parameters()), lr=0.01)


num_epochs = 5 # 1000
for i in range(num_epochs):
    for quantities in list_of_time_series:
        neural_net_loss = torch.tensor(0.0, requires_grad=True)
        h = torch.tensor([0.0, 0.0], requires_grad=True)
        h = h.view(1, 1, 2)
        prediction = quantities[0]
        for j in range(1, len(quantities)):
            neural_net_loss = neural_net_loss + (prediction - quantities[j]) ** 2
            current_quantity = torch.tensor(quantities[j])
            current_quantity = current_quantity.view(1, 1, 1)
            prediction, h = rnn(current_quantity, h)
            prediction = lin_layer(prediction)
        neural_net_loss = neural_net_loss/len(quantities)
        optimizer.zero_grad()
        neural_net_loss.backward()
        optimizer.step()


# prediction by neural net
for k in [0, 2]:  # , 3, 20, 100, 300]:
    quantities = list_of_time_series[k]
    h = torch.tensor([0.0, 0.0], requires_grad=True)
    h = h.view(1, 1, 2)
    prediction = quantities[0]
    for j in range(1, len(quantities)):
        current_quantity = torch.tensor(quantities[j])
        current_quantity = current_quantity.view(1, 1, 1)
        prediction, h = rnn(current_quantity, h)
        prediction = lin_layer(prediction)
        print("k=", k, "h=", h, "pred=", prediction)
    print(f"Prediction for articule {articule} on filial {filials_list[k]}: {reference_level * prediction}")

print("Finished!")
seconds = time.time() - start_time
hours = seconds / 3600
print(f"Program took {hours:.2f} hours")
