import numpy as np

import matplotlib.pyplot as plt
import scipy.integrate as integrate

#dx = .3
def my_integration(dx):
    x = np.arange(0, 100, dx)
    y = x**2 + 4*x + 5

    total_area = 0
    slice_area_list = []

    for idx, content in enumerate(x):
        y_content = content**2 + 4 * content + 5

        slice_area = y_content * dx
        slice_area_list.append(slice_area)
        total_area = total_area + slice_area
    return total_area

dx_list = np.arange(0.1,2,0.1)
integration_list = []

for content in dx_list:
    integration_list.append(my_integration(content))

average = np.array(integration_list).mean()



plt.plot(dx_list,integration_list)
    


def integrant(x):
    return x**2 + 4*x + 5


integration_by_scipy = integrate.quad(integrant, 0, 100)
print(total_area, integration_by_scipy)


# plt.hist(x,)
plt.figure()
plt.plot(x, y, "ro")
plt.plot(x, slice_area_list, "ko")
plt.grid(True)
