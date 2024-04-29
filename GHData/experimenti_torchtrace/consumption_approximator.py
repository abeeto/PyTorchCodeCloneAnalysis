import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import utils
from sklearn.linear_model import LinearRegression
from numpy import array 


# =============================================================================
# load data
data = pd.read_csv('./results/50Inserts_Records_10x10.csv')  # edit path

total_inserted_bytes = data['size'].sum()

max_size_disk = data['size_on_disk'].max()
min_size_disk = data['size_on_disk'].min()

print("max {0}".format(max_size_disk))
print("min {0}".format(min_size_disk))

total_disk_consumed = max_size_disk - min_size_disk 
data['size_on_disk'].max() - data['size_on_disk'].min()
friendly_inserted_bytes = utils.bytes_2_human_readable(total_inserted_bytes)
print(friendly_inserted_bytes)

data = data.sort_values(by=['time_stamp'])
x = data['size'].cumsum()
y = data['size_on_disk'].diff().fillna(0).cumsum()

print("Total inserted serialized data: {0}".format(utils.bytes_2_human_readable(total_inserted_bytes)))
print("Total disk consumption: {0}".format(utils.bytes_2_human_readable(total_disk_consumed)))


model = LinearRegression()
model.fit(x[:, np.newaxis], y)

print('Disk Consumption Equation: = {} x Inserted Data (MB) + {}'.format(model.coef_[0], model.intercept_))
size = float(input('enter size of Memory (bytes) to predict Disk Consumption = '))

pred = model.predict(size*1000000)
readable_estimate = utils.bytes_2_human_readable(pred)

print('Disk Consumption Estimate = {} '.format(readable_estimate))



