import os
import json
import matplotlib.pyplot as plt

def process_file(path:str):
    aux = path
    solver, i = path.replace('.json','').split('_')
    path = os.path.join('results', aux)
    with open(path, 'r') as file:
        data = json.load(file)
    data['solver'] = solver
    data['i'] = i

    return data

init = 10
end = 100000
step = end//15
iters = range(init, end, step)

root, _dir, files = next(os.walk('results'))
results = list(map(process_file, files))
solvers = set([data['solver'] for data in results])

fig, ax = plt.subplots(1,1, figsize=(12,8))
for sol in solvers:
    residuals = [data['residual'] for data in results if data['solver'] == sol]
    ax.plot(iters, residuals, label = sol)
    ax.scatter(iters, residuals)

ax.legend()
ax.set_xlabel('N Iters')
ax.set_ylabel('Log Residual')
plt.savefig('benchmark.jpg')
plt.close()
