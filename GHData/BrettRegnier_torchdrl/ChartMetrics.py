import matplotlib.pyplot as plt 
import torch    

fil = torch.load("models/checkpoints/CartPole_v4/checkpoint/episode_10_score_16.2.pt")
chart_metrics = fil['chart_metrics']
train_metrics = chart_metrics['train']
test_metrics = chart_metrics['test']

print(fil.keys())
print(chart_metrics.keys())
print(train_metrics.keys())
print(test_metrics['episode'])
plt.plot(train_metrics['episode'], train_metrics['score'], 'green')
plt.plot(train_metrics['episode'], train_metrics['steps'], 'purple')
plt.plot(train_metrics['episode'], train_metrics['loss'], 'blue')
plt.show()