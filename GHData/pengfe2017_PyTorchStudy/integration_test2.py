import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

# %%
# target function is:

def integrant(x):
    # return x**2 + 4*x + 5
    return np.cos(x)


def my_integration(integrant, dx):
    #dx = 0.3
    x = np.arange(0, 100, dx)
    y = integrant(x)

    total_area = 0
    slice_area_list = []

    for idx, content in enumerate(x):
        y_content = integrant(content)
        slice_area = y_content * dx
        slice_area_list.append(slice_area)
        total_area = total_area + slice_area

    total_area1 = np.array(slice_area_list).sum()
    return total_area


def my_integration_1(integrant, dx):
    #dx = 0.3
    x = np.arange(0, 100, dx)
    y = x**2 + 4*x + 5

    total_area = 0
    slice_area_list = []

    for idx, content in enumerate(x):
        x_k_next = content + dx
        y_content = integrant(x_k_next)
        slice_area = y_content * dx
        slice_area_list.append(slice_area)
        total_area = total_area + slice_area

    total_area1 = np.array(slice_area_list).sum()
    return total_area


def my_integration_2(integrant, dx):
    #dx = 0.01
    x = np.arange(0, 100, dx)
    y = x**2 + 4*x + 5

    total_area = 0
    slice_area_list = []

    for idx, content in enumerate(x):
        x_k_next = content + dx
        x_k = content
        y_content = (integrant(x_k_next) + integrant(x_k))/2
        slice_area = y_content * dx
        slice_area_list.append(slice_area)
        total_area = total_area + slice_area

    total_area1 = np.array(slice_area_list).sum()
    return total_area


# %%
integration_by_scipy = integrate.quad(integrant, 0, 100)

dx_list = np.arange(0.001, 1, 0.001)
integration_results_list = []

for idx, content in enumerate(dx_list):
    integration_results = my_integration_2(integrant, content)
    integration_results_list.append(integration_results)

# plt.figure()
print(integration_by_scipy[0], integration_results_list[0])
print("the difference between these two results are:",
      integration_by_scipy[0]-integration_results_list[0])
plt.plot(dx_list, integration_results_list, "r*")
plt.plot(dx_list, integration_results_list)
plt.plot(0, integration_by_scipy[0], "bo")
plt.grid(True)
