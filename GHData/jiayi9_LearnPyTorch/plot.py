# x, y plots

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10.0, 10.0, 0.1)
y = x**2

plt.plot(x, y)
plt.plot(np.array([1,2,3]), np.array([3,5,40]), "-o")

plt.xlabel("x values")
plt.ylabel("y values")



# x, y, z plots
import numpy as np
import matplotlib.pyplot as plt

x1, x2 = np.meshgrid(np.arange(-5.0, 5.0, 0.1), np.arange(-5.0, 5.0, 0.1))
y = x1**2 + x2**2
plt.contour(x1, x2, y, colors='#1f77b4')

path = [(0,0), (1,1), (2,3)]

list(zip(*path))

plt.plot(*zip(*path), "-o", color = 'red')
          
          
for x,y in zip(*zip(*path)):
    print(x,y)

for x,y in path:
    print(x,y)

path
list(zip(path))
list(zip(*path))
#list(*path)
#list(*zip(*path))




# Momentum is necessary for dealing with ravines


# understanding the different optimizers
# https://ruder.io/optimizing-gradient-descent/
