import numpy as np
import matplotlib.pyplot as plt
input = np.random.rand(50)
input = np.where(input < 0.1 ,1, 0)

out = []
w = 10
print(input.ndim)
for i in range(input.shape[0]):
    if i == 0:
        out.append(input[i])
    else:
        o = w*input[i] + 0.8 * out[-1]
        out.append(o)
plt.figure()
plt.subplot(1,2,1)
plt.title("Spike Train")
plt.plot(input)
plt.xlabel("time")

plt.subplot(1,2,2)
plt.title('Actual Value')
plt.plot(out)
plt.xlabel("time")
plt.tight_layout()
plt.show()
