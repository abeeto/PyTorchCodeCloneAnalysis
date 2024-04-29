import matplotlib.pyplot as plt
import numpy as np

a2c = []
with open('a2c_result.txt', 'r') as f:
    for row in f:
        a2c.append(float(row.strip()))

ppo = []
with open('ppo_result.txt', 'r') as f:
    for row in f:
        ppo.append(float(row.strip()))

t = np.arange(0, 2000, 1)
fig, ax = plt.subplots()
ax.plot(t, a2c, 'r-', label='a2c')
ax.plot(t, ppo, 'b-', label='ppo')
ax.set_title('Compare A2C with PPO in Stock trading')
ax.set_ylabel('Profits', loc='center', fontsize= 20)
plt.legend()
plt.show()