from pynet.layers.Dense import Dense
from pynet.layers.Output import Output
from pynet.network.Network import Network
from pynet.optimizers.RMSprop import RMSprop
from pynet.util.Util import normalize

import numpy as np
import torch as t
import keras.datasets.mnist

cuda0 = t.device('cuda:0')
(images, t_lables), (test_x, ttm) = keras.datasets.mnist.load_data()
temp = []


for ex in t_lables:
    o = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    o[ex] = 1
    temp.append(o)

images  = normalize(t.tensor(np.array([np.reshape(k, (28, 28)) for k in images]), dtype=t.float)).to(cuda0)
labels = t.tensor(temp, dtype=t.float).to(cuda0)



x = [Dense(shape=(28, 28), in_shape=(), optimizer=RMSprop(0.01, 20)),
     Dense(shape=(10, 10), in_shape=(28, 28), optimizer=RMSprop(0.01, 20)),
     Dense(shape=(7, 7), in_shape=(10, 10), optimizer=RMSprop(0.01, 20)),
     Dense(shape=(6, 6), in_shape=(7, 7), optimizer=RMSprop(0.01, 20)),
     Output(shape=(10,), in_shape=(6, 6), optimizer=RMSprop(0.01, 20))
     ]
model = Network(x)


while True:

    for i in range(60000):
        model.forward(images[i])
        model.backward(labels[i])
    model.evaluate(images, labels)
