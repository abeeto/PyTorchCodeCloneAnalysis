from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

x = [[1], [2], [3], [4]]
y = [[3], [6], [9], [12]]


def feed_forward(inputs, outputs, weights):
    out = np.dot(inputs, weights[0]) + weights[1]
    mean_squared_error = np.mean(np.square(out - outputs))
    return mean_squared_error


def update_weights(inputs, outputs, weights, lr):
    original_weights = deepcopy(weights)
    org_loss = feed_forward(inputs, outputs, original_weights)
    updated_weights = deepcopy(weights)
    for i, layer in enumerate(original_weights):
        for index, weight in np.ndenumerate(layer):
            temp_weights = deepcopy(weights)
            temp_weights[i][index] += 0.0001
            _loss_plus = feed_forward(inputs, outputs, temp_weights)
            grad = (_loss_plus - org_loss) / 0.0001
            updated_weights[i][index] -= grad * lr
            if i % 2 == 0:
                print('weight value:', np.round(original_weights[i][index], 2),
                      'original loss:', np.round(org_loss, 2),
                      'loss_plus:', np.round(_loss_plus, 2),
                      'gradient:', np.round(grad, 2),
                      'updated_weights:', np.round(updated_weights[i][index], 2))
    return updated_weights


W = [np.array([[0]], dtype=np.float32), np.array([[0]], dtype=np.float32)]

weight_value = []
for epx in range(10):
    W = update_weights(x, y, W, 0.01)
    weight_value.append(W[0][0][0])
print(W)

plt.plot(weight_value[:1000])
plt.title("Weights value over increasing epochs when learning rate is 0.01")
plt.xlabel('Epochs')
plt.ylabel('Weight value')
plt.show()
