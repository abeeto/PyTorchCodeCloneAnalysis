env = "Pendulum-v0"
hid = 256
l = 2
gamma = 0.99
seed = 0

seed = 0
steps_per_epoch = 4000
epochs = 100
replay_size = int(1e6)
polyak = 0.995
lr = 1e-3
alpha = 0.2
batch_size = 100
start_steps = 10000
update_after = 30
update_every = 10
num_test_episodes = 10
max_ep_len = 1000
save_freq = 1
eps = 0.0

# Parameters for Ray
n_cpu = 15
n_workers = 15


LOG_STD_MAX = 2
LOG_STD_MIN = -20