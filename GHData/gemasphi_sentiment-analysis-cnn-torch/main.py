from train import train
from test import test
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUM_EPOCHS = 10
LOG_INTERVAL = 10

MODEL_ARCH = {
    "conv_info": [
        {
            "in": 1,
            "out": 128,
            "size": 5,
            "pool": 2
        }, {
            "in": 128,
            "out": 128,
            "size": 5,
            "pool": 2
        }
    ],
    "linear_info": [ 
        {
            "out":128 
        }
    ],
    "output_info": {
        "in": 128, 
        "out": 5
    },
    "dropout" : {
        "use": True,
        "rate": 0.2
   }
} 
TRAINING_PARAMS = {
    "LR" : 0.05,
    "MM" : 0,
    "WD" : 0.05,
    "BATCH_SIZE": 50,
    "USE_EMBEDDINGS": True,
    "EMBEDDING_DIM": 300,
    "DEVICE" : torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

DATA_INFO = {
    "SD": "/data/trees",
    "GLOVE": "/data/glove.6B.300d.txt",
    "MODEL_NAME": "model"
}


def plot_loss(loss, disp_freq, name="loss.png"):
    iterations = np.array(range(0,len(loss)))*disp_freq
    fig, ax = plt.subplots()
    ax.set(xlabel='Iterations', ylabel='Loss')
    ax.plot(iterations,loss)
    ax.grid()
    fig.savefig(name)
    plt.show()
    

train_loss = train(DATA_INFO, MODEL_ARCH, TRAINING_PARAMS, NUM_EPOCHS, LOG_INTERVAL)
loss, acc = test(DATA_INFO, MODEL_ARCH, TRAINING_PARAMS, LOG_INTERVAL)


plot_loss(train_loss, LOG_INTERVAL)

data = {
        "mean_test_loss": loss,
        "mean_test_acc": acc,
    }

df  = pd.DataFrame(data, index=[0])
df.to_csv("test_results.cvs", header=True)