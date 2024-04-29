import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import torch
import torch.nn as nn
import sys
import time
import database_connection


start_time = time.time()

articule = 32485  # 361771 519429 97143 32547 32485 117
filial = 2016

start_date = "'2019-04-01'"
end_date = "'2019-08-31'"

# start_date = "'2018-02-03'"
# end_date = "'2018-08-31'"

# start_date = "'2018-05-03'"
# end_date = "'2018-11-03'"

conn = database_connection.DatabaseConnection()

df = conn.get_all_sales_by_articule_and_filial_with_residues(articule, filial, start_date, end_date)
quantities = df['quantity'].values

# now we want to find optimal alpha
beta = torch.tensor(0.5, requires_grad=True)
alpha = torch.sigmoid(beta)

rnn = nn.RNN(input_size=1, hidden_size=2, num_layers=1, batch_first=False)
lin_layer = nn.Linear(in_features=2, out_features=1)

optimizer = torch.optim.Adam([beta] + list(rnn.parameters()) + list(lin_layer.parameters()), lr=0.01)

num_epochs = 1000
for i in range(num_epochs):
    S = quantities[0]
    alpha = torch.sigmoid(beta)
    loss = torch.tensor(0.0, requires_grad=True)
    neural_net_loss = torch.tensor(0.0, requires_grad=True)

    """
    for j in range(1, len(quantities)):
        loss = loss + (S - quantities[j]) ** 2
        S = (1 - alpha) * S + alpha * quantities[j]
    """

    h = torch.tensor([0.0, 0.0], requires_grad=True)
    h = h.view(1, 1, 2)
    prediction = quantities[0]
    for j in range(1, len(quantities)):
        neural_net_loss = neural_net_loss + (prediction - quantities[j]) ** 2
        current_quantity = torch.tensor(quantities[j])
        current_quantity = current_quantity.view(1, 1, 1)
        prediction, h = rnn(current_quantity, h)
        prediction = lin_layer(prediction)

    # print("loss:", loss)
    # print("NN loss:", neural_net_loss)
    # sys.exit(-6)

    optimizer.zero_grad()
    # loss.backward()
    neural_net_loss.backward()
    optimizer.step()
alpha_from_pytorch = torch.sigmoid(beta)

# prediction by neural net
h = torch.tensor([0.0, 0.0], requires_grad=True)
h = h.view(1, 1, 2)
prediction = quantities[0]
for j in range(1, len(quantities)):
    current_quantity = torch.tensor(quantities[j])
    current_quantity = current_quantity.view(1, 1, 1)
    prediction, h = rnn(current_quantity, h)
    prediction = lin_layer(prediction)
print("Prediction by neural net:", prediction)

# using standard python library for computing alpha and prediction
model = SimpleExpSmoothing(quantities)
model_fit = model.fit()
alpha_from_holtwinters = model_fit.params['smoothing_level']
prediction_exp_smoothing = model_fit.predict()
print("Prediction by standard pythonian exp smoothing:", prediction_exp_smoothing)
print(f"Articule: {articule} filial: {filial} alpha from pytorch: {alpha_from_pytorch}  alpha from standard python module: {alpha_from_holtwinters}")

print("Finished!")
seconds = time.time() - start_time
hours = seconds / 3600
print(f"Program took {hours:.2f} hours")
