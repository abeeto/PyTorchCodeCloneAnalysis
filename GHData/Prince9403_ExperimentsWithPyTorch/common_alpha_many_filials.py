import time
import torch
import torch.nn as nn
import database_connection


start_time = time.time()

start_date = "'2018-04-01'"
end_date = "'2018-08-31'"

# start_date = "'2019-04-01'"
# end_date = "'2019-08-31'"

articule = 519429  # 361771 519429 97143 32547 32485 117

conn = database_connection.DatabaseConnection()
filials_set = conn.get_filials_for_articule(articule, start_date, end_date, min_days=40)
silpo_fora_trash_filials_set = conn.get_silpo_fora_trash_filials_set()
filials_set = filials_set.intersection(silpo_fora_trash_filials_set)

print("Number of filials:", len(filials_set))

list_of_time_series = []
list_of_promos = []

for filial in filials_set:
    df = conn.get_all_sales_by_articule_and_filial_with_residues(articule, filial, start_date, end_date)
    quantities = df['quantity'].values
    is_promo_day = df['is_promo_day'].values
    list_of_time_series.append(quantities)
    list_of_promos.append(is_promo_day)

start_time = time.time()

# now we want to find optimal alpha
beta = torch.tensor(0.5, requires_grad=True)
alpha = torch.sigmoid(beta)

optimizer = torch.optim.Adam([beta], lr=0.01)

num_epochs = 5
for i in range(num_epochs):
    for k in range(len(list_of_time_series)):
        quantities = list_of_time_series[k]
        is_promo_day = list_of_promos[k]
        S = quantities[0]
        alpha = torch.sigmoid(beta)
        loss = torch.tensor(0.0, requires_grad=True)
        for j in range(1, len(quantities)):
            if not is_promo_day[j]:
                loss = loss + (S - quantities[j]) ** 2
                S = (1 - alpha) * S + alpha * quantities[j]
        loss = loss / len(quantities)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
alpha = torch.sigmoid(beta)

print(f"Articule: {articule} alpha: {alpha}")

print("Finished!")
seconds = time.time() - start_time
hours = seconds / 3600
minutes = seconds / 60
print(f"Program took {hours:.2f} hours")
print(f"Program took {minutes:.2f} minutes")
print(f"Program took {seconds:.2f} seconds")
