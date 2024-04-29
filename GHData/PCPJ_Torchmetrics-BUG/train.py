import os
import yaml
import torch

import torchmetrics

from metrics.metric_factory import create_metrics
from utils import save_graph, device

print(torchmetrics.__version__)
_ = torch.manual_seed(42)

def main():

    training_conf = yaml.full_load(open('training_conf.yaml').read())
    metrics = create_metrics(training_conf['metrics'], training_conf)
    
    records = [[] for _ in metrics]

    for epoch in range(0, 2):  # loop over the dataset multiple times
        print("Epoch %d started!"%(epoch+1))
        
        for batch in range(0, 3):
            preds = torch.randn(100, 5, 28, 28).to(device)
            target = torch.randint(5, (100, 28, 28)).to(device)
            metrics(preds, target)
        

        accs = metrics.compute()
        print(accs)
        for i, metric_name in enumerate(metrics):
            records[i].append(accs[metric_name])

        for i, metric_name in enumerate(metrics):                        
            metrics[metric_name].save_graph("./", "model_name", epoch, records[i], records[i], [], [])

    print('Finished Training')

if __name__ == '__main__':
    main()
