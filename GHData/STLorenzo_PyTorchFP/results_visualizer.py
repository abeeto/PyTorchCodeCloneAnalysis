import pandas as pd
from pathlib import Path  # Path manipulation
from src.general_functions import *
import matplotlib.pyplot as plt

conf_filename = "/config/ImgConvNet_conf.json"
p_conf_data = read_conf("/config/Project_conf.json")
l_conf_data = read_conf(conf_filename)

base_path = Path(p_conf_data['base_path'])
data_base_path = base_path / p_conf_data['dirs']['data_dir']
created_data_path = data_base_path / l_conf_data['dirs']['created_data_dir']
logs_path = created_data_path / l_conf_data['dirs']['logs_dir']

# log_file = [x for x in os.listdir(logs_path) if 'optim' in x][0]
log_file = "prueba2.log"
print(log_file)
column_names = ['model', 'epoch', 'time', 'loss_function', 'optimizer', 'lr', 'batch_size',
                'train_acc', 'train_loss', 'val_acc', 'val_loss']

df = pd.read_csv(logs_path / log_file, names=column_names)

fig = plt.figure()
axes = []

max_i = 2
max_j = 2

for i in range(max_i):
    for j in range(max_j):
        if i == 0 and j == 0:
            axes.append(plt.subplot2grid((max_i, max_j), (i, j)))
        else:
            axes.append(plt.subplot2grid((max_i, max_j), (i, j), sharex=axes[0]))

metrics = ['train_acc', 'train_loss', 'val_acc', 'val_loss']
for ax, metric in zip(axes, metrics):
    ax.set_title(metric)
    for model in df['model'].unique():
        df_model = df[df['model'] == model]
        times = df_model['time']-df_model['time'].min()
        # times = df['epoch']
        ax.plot(times, df_model[metric], label=model)
    ax.legend()  # loc = location

plt.show()
print()
